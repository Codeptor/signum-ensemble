"""
Alerting module — multi-channel alert delivery for Signum trading bot.

Supports:
  - Telegram Bot API (RECOMMENDED — HTTPS port 443, instant, works everywhere)
  - Email via Resend HTTP API (port 443)
  - Email via SendGrid HTTP API (port 443)
  - Email via SMTP (fallback, blocked on DigitalOcean)
  - Webhook (Slack/Discord/generic)

Design principles:
  - Fire-and-forget: alerting failures NEVER propagate or mask real errors
  - Severity levels: INFO (heartbeats), WARNING (degraded), CRITICAL (action needed)
  - Rate limiting: prevent alert storms during cascading failures
  - Thread-safe: sends happen in background threads to avoid blocking

Usage:
    from python.monitoring.alerting import send_alert, AlertSeverity

    send_alert("Trade cycle completed: 8 fills", severity=AlertSeverity.INFO)
    send_alert("ML pipeline failed: timeout", severity=AlertSeverity.CRITICAL)
"""

import json
import logging
import os
import smtplib
import threading
import time
import urllib.request
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from zoneinfo import ZoneInfo

logger = logging.getLogger("signum.alerting")


# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------
class AlertSeverity(str, Enum):
    """Alert severity — determines email subject prefix and formatting."""

    INFO = "info"  # Heartbeats, successful trades
    WARNING = "warning"  # Degraded mode, stale data, caution regime
    CRITICAL = "critical"  # Crashes, kill switches, liquidations


# Severity → emoji-free prefix for email subjects
_SEVERITY_PREFIX = {
    AlertSeverity.INFO: "[Signum]",
    AlertSeverity.WARNING: "[Signum WARNING]",
    AlertSeverity.CRITICAL: "[Signum CRITICAL]",
}

# ---------------------------------------------------------------------------
# Configuration — all from environment variables
# ---------------------------------------------------------------------------

# Resend HTTP API (recommended — instant activation, uses port 443)
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "Signum Bot <onboarding@resend.dev>")

# SendGrid HTTP API (alternative — uses port 443)
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
SENDGRID_FROM_EMAIL = os.getenv("SENDGRID_FROM_EMAIL", "")

# Email (SMTP) — fallback when neither API is configured
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "")  # Sender address (defaults to SMTP_USER)
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() in ("true", "1", "yes")

# Recipients (shared by all transports)
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")  # Comma-separated

# Webhook (Slack/Discord/generic)
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")

# Telegram Bot API (recommended — HTTPS port 443, instant, free)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Rate limiting — max alerts per window to prevent storms
_RATE_LIMIT_WINDOW_SECS = 300  # 5-minute window
_RATE_LIMIT_MAX_ALERTS = 20  # Max alerts per window
_alert_timestamps: list[float] = []
_alert_lock = threading.Lock()

# Track last heartbeat to avoid spamming.
# Initialize to -inf so the first heartbeat always fires regardless of uptime.
_last_heartbeat_ts: float = float("-inf")
_HEARTBEAT_COOLDOWN_SECS = 3600  # At most one heartbeat per hour


def _is_resend_configured() -> bool:
    """Check if Resend API is configured."""
    return bool(RESEND_API_KEY and ALERT_EMAIL_TO)


def _is_sendgrid_configured() -> bool:
    """Check if SendGrid API is configured."""
    return bool(SENDGRID_API_KEY and SENDGRID_FROM_EMAIL and ALERT_EMAIL_TO)


def _is_smtp_configured() -> bool:
    """Check if SMTP email is configured."""
    return bool(SMTP_HOST and SMTP_USER and SMTP_PASSWORD and ALERT_EMAIL_TO)


def _is_email_configured() -> bool:
    """Check if any email transport is configured."""
    return _is_resend_configured() or _is_sendgrid_configured() or _is_smtp_configured()


def _is_telegram_configured() -> bool:
    """Check if Telegram Bot alerting is configured."""
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def _is_webhook_configured() -> bool:
    """Check if webhook alerting is configured."""
    return bool(ALERT_WEBHOOK_URL)


def _is_rate_limited() -> bool:
    """Check if we've exceeded the alert rate limit."""
    now = time.monotonic()
    with _alert_lock:
        # Prune old timestamps
        cutoff = now - _RATE_LIMIT_WINDOW_SECS
        _alert_timestamps[:] = [t for t in _alert_timestamps if t > cutoff]
        if len(_alert_timestamps) >= _RATE_LIMIT_MAX_ALERTS:
            return True
        _alert_timestamps.append(now)
        return False


# ---------------------------------------------------------------------------
# Email delivery — Resend (preferred) > SendGrid > SMTP (fallback)
# ---------------------------------------------------------------------------


def _send_email_resend(subject: str, body_text: str, body_html: str | None = None) -> None:
    """Send email via Resend HTTP API (https://resend.com/docs/api-reference)."""
    if not _is_resend_configured():
        return

    recipients = [r.strip() for r in ALERT_EMAIL_TO.split(",") if r.strip()]
    if not recipients:
        return

    body: dict = {
        "from": RESEND_FROM_EMAIL,
        "to": recipients,
        "subject": subject,
        "text": body_text,
    }
    if body_html:
        body["html"] = body_html

    payload = json.dumps(body).encode("utf-8")

    try:
        req = urllib.request.Request(
            "https://api.resend.com/emails",
            data=payload,
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=15)
        logger.debug(f"Resend email sent to {recipients}: {subject} (HTTP {resp.status})")
    except Exception:
        logger.warning("Failed to send Resend email", exc_info=True)


def _send_email_sendgrid(subject: str, body_text: str, body_html: str | None = None) -> None:
    """Send email via SendGrid v3 HTTP API (uses port 443, not blocked)."""
    if not _is_sendgrid_configured():
        return

    recipients = [r.strip() for r in ALERT_EMAIL_TO.split(",") if r.strip()]
    if not recipients:
        return

    content = [{"type": "text/plain", "value": body_text}]
    if body_html:
        content.append({"type": "text/html", "value": body_html})

    payload = json.dumps(
        {
            "personalizations": [{"to": [{"email": r} for r in recipients]}],
            "from": {
                "email": SENDGRID_FROM_EMAIL,
                "name": "Signum Bot",
            },
            "subject": subject,
            "content": content,
        }
    ).encode("utf-8")

    try:
        req = urllib.request.Request(
            "https://api.sendgrid.com/v3/mail/send",
            data=payload,
            headers={
                "Authorization": f"Bearer {SENDGRID_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=15)
        logger.debug(f"SendGrid email sent to {recipients}: {subject} (HTTP {resp.status})")
    except Exception:
        logger.debug("Failed to send SendGrid email", exc_info=True)


def _send_email_smtp(subject: str, body_text: str, body_html: str | None = None) -> None:
    """Send email via SMTP (fallback when SendGrid unavailable)."""
    if not _is_smtp_configured():
        return

    sender = SMTP_FROM or SMTP_USER
    recipients = [r.strip() for r in ALERT_EMAIL_TO.split(",") if r.strip()]
    if not recipients:
        return

    msg = MIMEMultipart("alternative")
    msg["From"] = f"Signum Bot <{sender}>"
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject

    msg.attach(MIMEText(body_text, "plain"))
    if body_html:
        msg.attach(MIMEText(body_html, "html"))

    try:
        if SMTP_USE_TLS:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15)
            server.ehlo()
            server.starttls()
            server.ehlo()
        else:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15)
            server.ehlo()

        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(sender, recipients, msg.as_string())
        server.quit()
        logger.debug(f"SMTP email sent to {recipients}: {subject}")
    except Exception:
        logger.debug("Failed to send SMTP email", exc_info=True)


def _send_email(subject: str, body_text: str, body_html: str | None = None) -> None:
    """Send email via best available transport (Resend > SendGrid > SMTP)."""
    if _is_resend_configured():
        _send_email_resend(subject, body_text, body_html)
    elif _is_sendgrid_configured():
        _send_email_sendgrid(subject, body_text, body_html)
    elif _is_smtp_configured():
        _send_email_smtp(subject, body_text, body_html)
    # else: no email transport configured — silently skip


def _format_email_html(message: str, severity: AlertSeverity) -> str:
    """Format alert as a simple HTML email body."""
    now = datetime.now(tz=ZoneInfo("America/New_York"))
    color = {
        AlertSeverity.INFO: "#2d7d46",
        AlertSeverity.WARNING: "#b8860b",
        AlertSeverity.CRITICAL: "#cc3333",
    }.get(severity, "#333333")

    # noqa: E501 — HTML email template; inline styles are intentionally long
    body_style = (
        "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; "
        "max-width: 600px; margin: 0 auto; padding: 20px;"
    )
    box_style = (
        f"border-left: 4px solid {color}; padding: 12px 16px; "
        "background: #f8f9fa; margin-bottom: 16px;"
    )
    label_style = (
        f"color: {color}; font-weight: 600; font-size: 12px; "
        "text-transform: uppercase; margin-bottom: 4px;"
    )
    msg_style = "color: #1a1a1a; font-size: 14px; line-height: 1.5; white-space: pre-wrap;"
    ts = now.strftime("%Y-%m-%d %H:%M:%S %Z")

    return f"""\
<html>
<body style="{body_style}">
  <div style="{box_style}">
    <div style="{label_style}">
      {severity.value}
    </div>
    <div style="{msg_style}">{message}</div>
  </div>
  <div style="color: #888; font-size: 11px; margin-top: 16px;">
    Signum Trading Bot &middot; {ts}
  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Telegram delivery
# ---------------------------------------------------------------------------

# Severity → Telegram prefix (emoji-free, monospace-friendly)
_TELEGRAM_SEVERITY_ICON = {
    AlertSeverity.INFO: "INFO",
    AlertSeverity.WARNING: "WARNING",
    AlertSeverity.CRITICAL: "CRITICAL",
}


def _send_telegram(message: str, severity: AlertSeverity = AlertSeverity.INFO) -> None:
    """Send message via Telegram Bot API (HTTPS port 443).

    Uses MarkdownV2 formatting with severity header. Falls back to plain text
    if MarkdownV2 fails (e.g., unescaped special characters).
    """
    if not _is_telegram_configured():
        return

    label = _TELEGRAM_SEVERITY_ICON.get(severity, "INFO")
    # Escape MarkdownV2 special chars in message body
    escaped = _escape_telegram_md(message)
    formatted = f"*\\[Signum {label}\\]*\n{escaped}"

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = json.dumps(
        {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": formatted,
            "parse_mode": "MarkdownV2",
        }
    ).encode("utf-8")

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=10)
        logger.debug(f"Telegram alert sent (HTTP {resp.status}): {message[:60]}")
    except Exception:
        # Fallback: try without MarkdownV2 in case of escaping issues
        try:
            plain = f"[Signum {label}]\n{message}"
            payload_plain = json.dumps(
                {
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": plain,
                }
            ).encode("utf-8")
            req2 = urllib.request.Request(
                url,
                data=payload_plain,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req2, timeout=10)
            logger.debug(f"Telegram alert sent (plain fallback): {message[:60]}")
        except Exception:
            logger.debug("Failed to send Telegram alert", exc_info=True)


def _escape_telegram_md(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2 format."""
    # These characters must be escaped with a preceding backslash
    special = r"_*[]()~`>#+-=|{}.!"
    result = []
    for ch in text:
        if ch in special:
            result.append(f"\\{ch}")
        else:
            result.append(ch)
    return "".join(result)


# ---------------------------------------------------------------------------
# Webhook delivery
# ---------------------------------------------------------------------------


def _send_webhook(message: str) -> None:
    """POST JSON payload to webhook URL (Slack/Discord compatible)."""
    if not _is_webhook_configured():
        return
    try:
        payload = json.dumps({"text": message}).encode("utf-8")
        req = urllib.request.Request(
            ALERT_WEBHOOK_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        logger.debug("Failed to send webhook alert", exc_info=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def send_alert(
    message: str,
    severity: AlertSeverity = AlertSeverity.WARNING,
    *,
    subject: str | None = None,
    bypass_rate_limit: bool = False,
) -> None:
    """Send an alert through all configured channels.

    This is the single entry point for all alerting. It:
      1. Checks rate limits (unless bypassed for CRITICAL)
      2. Sends webhook (synchronous, fast)
      3. Sends email (background thread, ~2-5s)

    Never raises — alerting failures are swallowed and logged at DEBUG level.

    Parameters
    ----------
    message : str
        The alert message body.
    severity : AlertSeverity
        INFO, WARNING, or CRITICAL. Affects email subject and formatting.
    subject : str, optional
        Custom email subject. Auto-generated from severity + first line if omitted.
    bypass_rate_limit : bool
        Skip rate limiting. Use for CRITICAL alerts that must always deliver.
    """
    try:
        # Rate limiting (CRITICAL always bypasses)
        if severity == AlertSeverity.CRITICAL:
            bypass_rate_limit = True

        if not bypass_rate_limit and _is_rate_limited():
            logger.debug(
                f"Alert rate-limited (>{_RATE_LIMIT_MAX_ALERTS}/{_RATE_LIMIT_WINDOW_SECS}s)"
            )
            return

        # Build subject line
        prefix = _SEVERITY_PREFIX.get(severity, "[Signum]")
        if subject is None:
            # Use first line of message, truncated
            first_line = message.split("\n")[0][:80]
            subject = f"{prefix} {first_line}"
        else:
            subject = f"{prefix} {subject}"

        # Telegram (synchronous — fast at ~100-300ms via HTTPS)
        _send_telegram(message, severity)

        # Webhook (synchronous — fast enough at ~100ms)
        _send_webhook(message)

        # Email (background thread to avoid blocking trading logic)
        if _is_email_configured():
            body_html = _format_email_html(message, severity)
            thread = threading.Thread(
                target=_send_email,
                args=(subject, message, body_html),
                daemon=True,
            )
            thread.start()

    except Exception:
        # Nuclear fallback — alerting must never crash anything
        logger.debug("send_alert failed entirely", exc_info=True)


def send_heartbeat(
    message: str,
    *,
    force: bool = False,
) -> None:
    """Send an INFO heartbeat alert, rate-limited to one per hour.

    Use for periodic "all is well" signals so silence = problem.
    """
    global _last_heartbeat_ts
    with _alert_lock:
        now = time.monotonic()
        if not force and (now - _last_heartbeat_ts) < _HEARTBEAT_COOLDOWN_SECS:
            return
        _last_heartbeat_ts = now
    send_alert(message, severity=AlertSeverity.INFO)


def send_trade_summary(
    filled: int,
    partial: int,
    failed: int,
    total: int,
    positions: dict[str, float] | None = None,
    equity: float | None = None,
) -> None:
    """Send a structured trade cycle summary.

    Aggregates fill results into a single actionable alert.
    """
    lines = [
        f"Trade cycle completed: {filled} filled, "
        f"{partial} partial, {failed} failed / {total} submitted"
    ]
    if equity is not None:
        lines.append(f"Portfolio equity: ${equity:,.2f}")
    if positions:
        lines.append(f"Holdings ({len(positions)}):")
        for ticker, weight in sorted(positions.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  {ticker}: {weight:.1%}")

    severity = AlertSeverity.INFO if failed == 0 else AlertSeverity.WARNING
    send_alert(
        "\n".join(lines),
        severity=severity,
        subject="Trade cycle summary",
    )


def get_alerting_status() -> dict:
    """Return current alerting configuration status (for /healthz)."""
    transport = "none"
    if _is_resend_configured():
        transport = "resend"
    elif _is_sendgrid_configured():
        transport = "sendgrid"
    elif _is_smtp_configured():
        transport = "smtp"

    return {
        "telegram_configured": _is_telegram_configured(),
        "email_configured": _is_email_configured(),
        "email_transport": transport,
        "webhook_configured": _is_webhook_configured(),
        "smtp_host": SMTP_HOST or None,
        "sendgrid_configured": _is_sendgrid_configured(),
        "resend_configured": _is_resend_configured(),
        "recipients": (
            [r.strip() for r in ALERT_EMAIL_TO.split(",") if r.strip()] if ALERT_EMAIL_TO else []
        ),
        "rate_limit": (f"{_RATE_LIMIT_MAX_ALERTS}/{_RATE_LIMIT_WINDOW_SECS}s"),
    }
