# nimamanafcom

Private personal dashboard accessible **ONLY via Tailscale network**. Provides terminal access and Ralph AI project management.

## Security Model

This application is designed for private use on a Tailscale network. It is **never** exposed to the public internet.

```
┌─────────────────────────────────────────────────────────────┐
│                    PUBLIC INTERNET                          │
│                         ❌ NO ACCESS                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   TAILSCALE NETWORK                         │
│              (Only your devices can connect)                │
│                                                             │
│   Phone ──┐                                                 │
│   Laptop ─┼──► https://machine.tailnet.ts.net               │
│   Other  ─┘                                                 │
└─────────────────────────────────────────────────────────────┘
```

**Security Layers:**
1. **Tailscale network** - Primary barrier (only your authenticated devices)
2. **Password authentication** - Secondary verification
3. **Signed session cookies** - Prevents tampering (itsdangerous)

## Prerequisites

- Python 3.10+
- Tailscale installed and configured on the server
- The server must be part of your Tailscale network

## Quick Start

### 1. Clone and Setup

```bash
cd /home/nima/code/web/personal/nimamanafcom
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Create .env Configuration

```bash
cp .env.example .env
# Edit .env with your values
```

Required environment variables:

```
SECRET_KEY=your-random-secret-key-here
PASSWORD_HASH=your-bcrypt-password-hash
```

### 3. Generate Password Hash

```bash
python -c "import bcrypt; print(bcrypt.hashpw(b'your-password-here', bcrypt.gensalt()).decode())"
```

Copy the output to `PASSWORD_HASH` in your `.env` file.

### 4. Test Locally

```bash
source venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Visit http://127.0.0.1:8000 in your browser.

## Tailscale Configuration

### Enable Tailscale Serve

This exposes the app to your Tailscale network (but NOT the public internet).

```bash
# Enable serve for port 8000
tailscale serve --bg 8000

# Check status
tailscale serve status
```

Your app will be available at:
```
https://<machine-name>.<tailnet>.ts.net
```

For example: `https://server.tail12345.ts.net`

### Important: Do NOT Use Funnel

**WARNING:** Do NOT use `tailscale funnel`. Funnel exposes your app to the public internet, which defeats the entire security model.

```bash
# NEVER DO THIS:
# tailscale funnel 8000  ← This would make it PUBLIC
```

### HTTPS

Tailscale automatically provides HTTPS certificates for your `.ts.net` domain. No additional configuration needed.

## Systemd Service (Auto-Start)

### Install the Service

```bash
mkdir -p ~/.config/systemd/user/
cp deploy/nimamanafcom.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable nimamanafcom
systemctl --user start nimamanafcom
```

### Service Commands

```bash
# Check status
systemctl --user status nimamanafcom

# View logs
journalctl --user -u nimamanafcom -f

# Restart
systemctl --user restart nimamanafcom

# Stop
systemctl --user stop nimamanafcom
```

### Enable Lingering (Optional)

To keep user services running after logout:

```bash
sudo loginctl enable-linger $USER
```

## Features

### Landing Page (/)
- Public info page (no password required)
- Accessible once you're on Tailscale

### Dashboard (/dashboard)
- Navigation hub
- System info display
- Links to Terminal and Ralph

### Terminal (/terminal)
- Full shell access via xterm.js
- WebSocket-based PTY
- Resize support
- Copy/paste works

### Ralph (/ralph)
- AI project management dashboard
- View/manage projects
- Start/stop Ralph sessions
- Stream logs

## Project Structure

```
nimamanafcom/
├── app/
│   ├── main.py              # FastAPI application
│   ├── auth.py              # Password & session management
│   ├── config.py            # Settings from .env
│   ├── middleware.py        # Session validation middleware
│   ├── terminal.py          # WebSocket PTY handler
│   ├── ralph_integration.py # Ralph dashboard integration
│   ├── routers/             # (empty - routes in main.py)
│   ├── templates/           # Jinja2 HTML templates
│   └── static/              # CSS files
├── deploy/
│   └── nimamanafcom.service # systemd service file
├── .env                     # Configuration (create from .env.example)
├── .env.example             # Example configuration
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Testing Checklist

After setup, verify:

- [ ] App starts without errors locally
- [ ] Can access via Tailscale from another device (phone, laptop)
- [ ] Cannot access from public internet
- [ ] Login works with your password
- [ ] Dashboard shows system info
- [ ] Terminal connects and runs commands
- [ ] Ralph dashboard loads and shows projects
- [ ] Session expires after timeout (default 24h)
- [ ] Logout clears session

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| SECRET_KEY | Yes | - | Random string for signing cookies |
| PASSWORD_HASH | Yes | - | Bcrypt hash of your password |
| SESSION_TIMEOUT_HOURS | No | 24 | Session expiry in hours |
| HOST | No | 127.0.0.1 | Bind address (keep as localhost) |
| PORT | No | 8000 | Port number |

## Troubleshooting

### "Session expired" immediately after login
- Check that SECRET_KEY is set and unchanged
- Verify system clock is correct

### Terminal doesn't connect
- Check WebSocket connection in browser devtools
- Ensure PTY process spawns (check logs)

### Ralph dashboard shows errors
- Verify Ralph app is at /home/nima/code/ralph
- Check that Ralph's dependencies are available

### Tailscale serve not working
- Run `tailscale status` to check connection
- Verify serve with `tailscale serve status`
- Check firewall isn't blocking port 8000

## Dependencies

Key dependencies:
- FastAPI - Web framework
- Uvicorn - ASGI server
- Jinja2 - Templates
- itsdangerous - Signed cookies
- bcrypt - Password hashing
- ptyprocess - Terminal PTY
- python-dotenv - Environment config

See `requirements.txt` for full list.
