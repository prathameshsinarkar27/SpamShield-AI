"""
app/services/gmail_service.py
Production Gmail integration:
  - OAuth 2.0 auth with auto-refresh
  - Rate limiting (token bucket)
  - Full email metadata fetch
  - Spam automation: label + move to spam folder
  - Retry logic on transient API errors
"""

import os
import re
import base64
import time
import threading
from functools import wraps
from app.utils.logger import get_logger

logger = get_logger("spamshield.gmail")

# ── OAuth scopes ──────────────────────────────────────────────────────────────
# Using modify scope so we can label/move spam emails
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
]

BASE_DIR       = os.path.dirname(__file__)
ROOT_DIR       = os.path.join(BASE_DIR, "..", "..")
TOKEN_PATH      = os.path.join(ROOT_DIR, "token.json")
CREDENTIALS_PATH = os.path.join(ROOT_DIR, "credentials.json")


# ═════════════════════════════════════════════════════════════════════════════
# RATE LIMITER (token bucket — 60 requests/minute)
# ═════════════════════════════════════════════════════════════════════════════
class _TokenBucket:
    def __init__(self, rate: float = 60.0, capacity: float = 60.0):
        self._rate     = rate          # tokens per second (60/min → 1/s)
        self._capacity = capacity
        self._tokens   = capacity
        self._lock     = threading.Lock()
        self._last     = time.monotonic()

    def consume(self, tokens: float = 1.0) -> bool:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self._capacity,
                self._tokens + (now - self._last) * (self._rate / 60.0),
            )
            self._last = now
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def wait_and_consume(self, tokens: float = 1.0, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.consume(tokens):
                return
            time.sleep(0.1)
        raise TimeoutError("Gmail rate-limit timeout — try again in a moment")


_rate_limiter = _TokenBucket(rate=60, capacity=60)


def _rate_limited(fn):
    """Decorator that enforces the token-bucket rate limit before each call."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        _rate_limiter.wait_and_consume()
        return fn(*args, **kwargs)
    return wrapper


# ═════════════════════════════════════════════════════════════════════════════
# AUTH
# ═════════════════════════════════════════════════════════════════════════════
def _get_service():
    """Build and return an authenticated Gmail API service object."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if os.path.exists(TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        except Exception as exc:
            logger.warning("Could not load token.json: %s", exc)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                logger.info("Gmail token refreshed")
            except Exception as exc:
                logger.error("Token refresh failed: %s", exc)
                creds = None

        if creds is None:
            if not os.path.exists(CREDENTIALS_PATH):
                raise FileNotFoundError(
                    "credentials.json not found. Download it from Google Cloud Console."
                )
            flow  = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
            logger.info("New Gmail OAuth flow completed")

        with open(TOKEN_PATH, "w") as fh:
            fh.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


# ═════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _get_header(headers: list, name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _decode_body(part: dict) -> str:
    data = part.get("body", {}).get("data", "")
    if data:
        try:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        except Exception:
            pass
    return ""


def _extract_body(payload: dict) -> str:
    """Recursively extract the best plain-text body from a MIME payload."""
    mime = payload.get("mimeType", "")
    if mime == "text/plain":
        return _decode_body(payload)
    if mime == "text/html":
        raw = _decode_body(payload)
        return re.sub(r"<[^>]+>", " ", raw).strip()
    for part in payload.get("parts", []):
        result = _extract_body(part)
        if result and result.strip():
            return result
    return ""


def _parse_date(date_str: str) -> str:
    if not date_str:
        return ""
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(date_str)
        return dt.strftime("%d %b %Y, %I:%M %p")
    except Exception:
        return date_str[:25]


def _retry(fn, retries: int = 3, backoff: float = 1.0):
    """Simple retry wrapper for transient API errors."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = backoff * (2 ** attempt)
            logger.warning("API error (attempt %d/%d), retrying in %.1fs: %s",
                           attempt + 1, retries, wait, exc)
            time.sleep(wait)


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════
@_rate_limited
def fetch_emails(max_results: int = 20, query: str = "in:inbox") -> list[dict]:
    """
    Fetch emails from Gmail with full metadata.

    Args:
        max_results: Max number of emails (capped at 50).
        query:       Gmail search query string.

    Returns:
        List of email dicts: id, subject, sender, sender_short,
        date, snippet, body (≤2000 chars), is_unread.
    """
    max_results = min(max_results, 50)
    logger.info("fetch_emails — query=%r, max=%d", query, max_results)

    service = _get_service()
    q = query or "in:inbox"

    try:
        resp = _retry(lambda: service.users().messages().list(
            userId="me", q=q, maxResults=max_results
        ).execute())
    except Exception as exc:
        logger.error("Gmail list failed: %s", exc)
        raise

    messages = resp.get("messages", [])
    emails   = []

    for msg in messages:
        try:
            full = _retry(lambda mid=msg["id"]: service.users().messages().get(
                userId="me", id=mid, format="full"
            ).execute())

            headers      = full.get("payload", {}).get("headers", [])
            subject      = _get_header(headers, "Subject") or "(No Subject)"
            sender       = _get_header(headers, "From")    or "Unknown"
            date_str     = _get_header(headers, "Date")    or ""
            snippet      = full.get("snippet", "")
            body         = _extract_body(full.get("payload", {})).strip() or snippet
            sender_short = re.sub(r"\s*<[^>]+>", "", sender).strip() or sender
            is_unread    = "UNREAD" in full.get("labelIds", [])

            emails.append({
                "id":           msg["id"],
                "subject":      subject[:80],
                "sender":       sender,
                "sender_short": sender_short[:30],
                "date":         _parse_date(date_str),
                "snippet":      snippet[:120],
                "body":         body[:2000],
                "is_unread":    is_unread,
            })
        except Exception as exc:
            logger.warning("Skipping email %s: %s", msg.get("id"), exc)

    logger.info("Fetched %d emails", len(emails))
    return emails


@_rate_limited
def mark_as_spam(message_id: str) -> dict:
    """
    Automatically:
      1. Add SPAM label to the message
      2. Remove INBOX label (moves to spam folder)

    Returns result dict with success flag.
    """
    logger.info("Marking email %s as spam", message_id)
    service = _get_service()

    try:
        result = _retry(lambda: service.users().messages().modify(
            userId="me",
            id=message_id,
            body={
                "addLabelIds":    ["SPAM"],
                "removeLabelIds": ["INBOX"],
            },
        ).execute())
        logger.info("Email %s moved to spam", message_id)
        return {"success": True, "message_id": message_id, "labels": result.get("labelIds", [])}
    except Exception as exc:
        logger.error("Failed to mark %s as spam: %s", message_id, exc)
        return {"success": False, "error": str(exc)}


@_rate_limited
def get_label_list() -> list[dict]:
    """Return all Gmail labels for the authenticated user."""
    service = _get_service()
    try:
        resp   = _retry(lambda: service.users().labels().list(userId="me").execute())
        labels = resp.get("labels", [])
        logger.debug("Fetched %d labels", len(labels))
        return [{"id": l["id"], "name": l["name"]} for l in labels]
    except Exception as exc:
        logger.error("Failed to fetch labels: %s", exc)
        return []
