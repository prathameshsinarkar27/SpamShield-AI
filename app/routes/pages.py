"""
app/routes/pages.py
Blueprint for the HTML pages (multi-page UI).

"""

from flask import Blueprint, render_template, redirect, url_for

pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def index():
    """Redirect the root URL to the detector page."""
    return redirect(url_for("pages.detect"))


@pages_bp.route("/detect")
def detect():
    """Dataset / Custom text input + prediction result + LIME explanation."""
    return render_template("detector.html", active_page="detect")

