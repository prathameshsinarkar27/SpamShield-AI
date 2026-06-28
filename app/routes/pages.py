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


@pages_bp.route("/dashboard")
def dashboard():
    """Model performance dashboard: stats, charts, confusion matrices."""
    return render_template("dashboard.html", active_page="dashboard")


@pages_bp.route("/gmail")
def gmail_page():
    """Gmail inbox scanning + its own prediction result panel."""
    return render_template("gmail.html", active_page="gmail")


@pages_bp.route("/analytics")
def analytics():
    """NLP pipeline overview + DNN architecture diagram."""
    return render_template("analytics.html", active_page="analytics")
