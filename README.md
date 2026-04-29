# em-core

Shared Python package powering both the **EM Dashboard** (Flask, Render) and the
**TripWire Telegram bot**. Contains the single source of truth for:

- Polygon.io REST client (`em_core.polygon`)
- Implied volatility / expected-move math (`em_core.iv`)
- Daily / weekly / monthly level locking (`em_core.locking`, 12am ET)
- 700-ticker universe + dashboard groups (`em_core.universe`)
- SQLite storage: watchlists, level cache, alert dedupe, earnings cache (`em_core.storage`)
- Earnings calendar fetch + session-window helper (`em_core.earnings`)

## Install (editable, for local dev)

```powershell
pip install -e C:\Users\Kurta\em-core
```

Run the same command from both the `tripwire-bot` and `em-dashboard` virtualenvs so
edits to em-core are picked up instantly in both apps.

## Layout

```
em-core/
  pyproject.toml
  README.md
  src/em_core/
    __init__.py
    polygon.py
    iv.py
    locking.py
    universe.py
    storage.py
    earnings.py
```
