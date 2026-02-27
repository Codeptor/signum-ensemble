# Signum Trading Bot - VPS Deployment Guide

This guide walks you through deploying the Signum trading bot on a VPS (Virtual Private Server) for paper trading.

## Current Deployment

The bot is live on DigitalOcean:
- **VPS:** 209.38.122.78 (2 vCPU, 4 GB RAM, 80 GB disk, Ubuntu 22.04, Bangalore)
- **Services:** `signum-bot` (trading) + `signum-dashboard` (web UI on port 8050)
- **Dashboard:** Public at `https://dashboard.bhanueso.dev` (nginx + Let's Encrypt)
- **Telegram:** Alerts + interactive commands via `@sigum_paperbot`
- **SSH:** `ssh -i deploy/signum_ed25519 root@209.38.122.78`
- **Resource usage:** ~530 MB RAM (14%), 5.2 GB disk (7%), CPU idle at 1-3%

## Prerequisites

- VPS with Ubuntu 20.04+ or Debian 11+
- At least 2GB RAM, 10GB disk space (4GB RAM recommended for ML training)
- Root or sudo access
- Alpaca Paper Trading API credentials
- Telegram bot token (from @BotFather) for alerts

## Quick Start

### 1. Provision Your VPS

Recommended providers:
- **DigitalOcean**: $6-12/month (2GB RAM)
- **Linode**: $5-10/month (1-2GB RAM)
- **Hetzner**: €4-8/month (2-4GB RAM)
- **AWS/GCP/Azure**: Free tier eligible

### 2. Upload the Code

```bash
# On your local machine, upload the code to VPS
scp -r /path/to/quant user@your-vps-ip:/tmp/quant

# SSH into your VPS
ssh user@your-vps-ip

# Move to final location
sudo mv /tmp/quant /opt/signum
sudo chown -R root:root /opt/signum
```

### 3. Run Setup Script

```bash
cd /opt/signum
sudo bash deploy/setup-vps.sh
```

This will:
- Install Python, uv, and dependencies
- Create `signum` user
- Set up directories and permissions
- Install systemd service
- Configure log rotation

### 4. Configure Environment

```bash
# Edit the environment file
sudo nano /opt/signum/.env
```

Add your credentials:

```bash
# REQUIRED: Alpaca Paper Trading API
ALPACA_API_KEY=your_paper_api_key_here
ALPACA_API_SECRET=your_paper_api_secret_here

# RECOMMENDED: Telegram Bot (alerts + commands)
# Create bot via @BotFather, message it, get chat_id from getUpdates API
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Optional: Alert Webhook (Slack/Discord)
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Optional: Email alerts (Resend or SendGrid — SMTP is blocked on most cloud VPS)
RESEND_API_KEY=
RESEND_FROM_EMAIL=Signum Bot <onboarding@resend.dev>
SENDGRID_API_KEY=
SENDGRID_FROM_EMAIL=
ALERT_EMAIL_TO=your_email@gmail.com

# Risk Parameters (defaults are safe for paper trading)
MAX_POSITION_WEIGHT=0.30
STOP_LOSS_PCT=0.05
TAKE_PROFIT_PCT=0.15
MAX_DRAWDOWN_LIMIT=0.15
```

**Get your API keys from:** https://app.alpaca.markets/paper/dashboard

**Note:** DigitalOcean (and most cloud VPS) blocks outbound SMTP (ports 25/465/587). Use Telegram or HTTP-based email APIs (Resend/SendGrid on port 443) instead.

### 5. Test Before Starting Service

```bash
# Switch to bot user
sudo su - signum

# Test API connectivity
cd /opt/signum
source .venv/bin/activate
python -c "
from python.brokers.alpaca_broker import AlpacaBroker
import os
broker = AlpacaBroker(
    paper_trading=True,
    api_key=os.getenv('ALPACA_API_KEY'),
    api_secret=os.getenv('ALPACA_API_SECRET')
)
print('Connected:', broker.connect())
print('Account:', broker.get_account())
"

# Exit back to root
exit
```

### 6. Start the Bot

```bash
# Enable and start service
sudo systemctl enable signum-bot
sudo systemctl start signum-bot

# Check status
sudo systemctl status signum-bot

# View logs
sudo journalctl -u signum-bot -f
```

## Monitoring

### Telegram Bot (Primary — from your phone)

Send these commands to your Telegram bot:

| Command | What |
|---------|------|
| `/status` | Bot state, regime, account overview |
| `/positions` | Current holdings with P&L |
| `/equity` | Portfolio value, cash, buying power |
| `/regime` | VIX, SPY drawdown, exposure |
| `/health` | System health check |
| `/trades` | Recent trade info |
| `/logs` | Last 20 log lines |
| `/help` | All commands |

The bot also sends automatic alerts (18 events) for startup, trade summaries, regime changes, crashes, and kill switches.

### Server-Side

```bash
# Service status
sudo systemctl status signum-bot

# Recent logs
sudo journalctl -u signum-bot --since "1 hour ago"

# Live log tail
sudo tail -f /var/log/signum/bot.log

# Error logs
sudo tail -f /var/log/signum/bot-error.log

# Health check
curl http://localhost:8050/healthz
```

### Dashboard

Public at `https://dashboard.bhanueso.dev` or via SSH tunnel:
```bash
ssh -L 8050:localhost:8050 -i deploy/signum_ed25519 root@209.38.122.78
# Then open http://localhost:8050
```

### JSON API

12 endpoints at `http://localhost:8050/api/...` — see README for full list.

### View Trading Activity

```bash
# Check state file
cat /opt/signum/data/bot_state.json

# Quick position check via API
curl -s http://localhost:8050/api/positions | python3 -m json.tool
```

## Management Commands

### Stop the Bot

```bash
sudo systemctl stop signum-bot
```

### Restart the Bot

```bash
sudo systemctl restart signum-bot
```

### Disable Auto-Start

```bash
sudo systemctl disable signum-bot
```

### Update the Bot

```bash
# Pull latest code
cd /opt/signum
sudo git pull

# Update dependencies
sudo -u signum uv pip install -e ".[broker]"

# Restart service
sudo systemctl restart signum-bot
```

## Safety Features

The bot includes multiple safety mechanisms:

1. **Paper Trading Mode**: Only paper trading is enabled by default (`LIVE_TRADING=true` required for real money)
2. **Duplicate Trade Guard**: Prevents trading twice on the same day (checks Alpaca order history)
3. **Position Limits**: Max 30% per position (configurable via `.env`)
4. **ATR-Based Stops**: 2x ATR stop-loss, 3x ATR take-profit (adapts to each stock's volatility)
5. **Max Drawdown Kill Switch**: Liquidates all positions if portfolio drops 15%
6. **Regime Detection**: VIX/SPY drawdown monitoring — reduces to 50% exposure in caution, liquidates in halt
7. **Crash Recovery**: `systemctl Restart=on-failure` with backoff
8. **State Persistence**: Atomic writes to `data/bot_state.json` across restarts
9. **Telegram Alerts**: 18 event types with severity levels, CRITICAL bypasses rate limits
10. **Log Rotation**: Python `RotatingFileHandler` (10MB x 5) + system `logrotate` (weekly, 12 weeks compressed)

## Troubleshooting

### Bot Won't Start

```bash
# Check for config errors
sudo journalctl -u signum-bot --no-pager | head -50

# Verify environment file
sudo -u signum cat /opt/signum/.env | grep -E "^(ALPACA|MAX)"

# Test manually
sudo -u signum /opt/signum/.venv/bin/python /opt/signum/examples/live_bot.py
```

### API Connection Issues

```bash
# Test API connectivity
sudo -u signum /opt/signum/.venv/bin/python -c "
from python.brokers.alpaca_broker import AlpacaBroker
import os
broker = AlpacaBroker(paper_trading=True, api_key=os.getenv('ALPACA_API_KEY'), api_secret=os.getenv('ALPACA_API_SECRET'))
print('Connect:', broker.connect())
print('Account:', broker.get_account())
print('Clock:', broker.get_clock())
"
```

### High Memory Usage

```bash
# Check memory usage
sudo systemctl status signum-bot

# View resource limits
sudo systemctl show signum-bot | grep -E "(Memory|CPU)"

# Restart if needed
sudo systemctl restart signum-bot
```

### Disk Space Issues

```bash
# Check log size
sudo du -sh /var/log/signum/

# Force log rotation
sudo logrotate -f /etc/logrotate.d/signum

# Clean old state files
sudo find /opt/signum/data -name "*.json" -mtime +30 -delete
```

## Security Checklist

- [ ] API keys stored in `.env` file (not in code)
- [ ] `.env` file has 600 permissions (owner read/write only)
- [ ] Bot runs as non-root user (`signum`)
- [ ] Firewall allows only necessary ports (SSH, HTTPS)
- [ ] Server has automatic security updates enabled
- [ ] Logs are rotated daily
- [ ] SSH key authentication (not password)
- [ ] Fail2ban installed for brute force protection

## Cost Estimation

**VPS Costs:**
- DigitalOcean: $6-12/month
- Linode: $5-10/month
- Hetzner: €4-8/month

**Alpaca Paper Trading:**
- Free (no commissions or fees)

**Optional:**
- Alert webhooks: Free (Slack/Discord)
- Backup storage: $1-5/month

## Next Steps

1. **Monitor for 1 week**: Watch logs, verify trades execute correctly
2. **Paper trade for 1 month**: Validate strategy performance
3. **Graduate to live trading** (if desired): Change to Alpaca Live API keys
4. **Scale up**: Consider multiple bots, different strategies

## Support

- Check logs: `sudo journalctl -u signum-bot -f`
- Review code: All source is in `/opt/signum/`
- Configuration: Edit `/opt/signum/.env`
- Service control: `sudo systemctl {start|stop|restart|status} signum-bot`

---

**IMPORTANT**: This bot is configured for **PAPER TRADING ONLY** by default. To switch to live trading:
1. Set `LIVE_TRADING=true` in `.env` (do NOT edit source code)
2. Use Alpaca Live API keys (not Paper keys)
3. **WARNING**: Real money will be at risk. Paper trade for 3+ months first.
