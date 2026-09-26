# Security Policy

## Reporting a vulnerability

Please do not publish credentials, private data, or detailed exploit steps in a
public issue.

Use GitHub's private vulnerability reporting if it is enabled for this
repository. Otherwise contact `c0k0n` through
[the GitHub profile](https://github.com/c0k0n), with the affected version,
reproduction steps, likely impact, and safe supporting evidence.

## Scope

This is an exploratory learning project and is **not financial advice**. It
downloads public market data and trains a small model locally. There is no
authentication, no user accounts, and no network service of its own.

One thing does persist: forecasts are written to a local SQLite file
(`journal.db`) so they can be graded once their dates pass. It holds tickers,
timestamps and prices only — no credentials — and it is gitignored. On Streamlit
Community Cloud the filesystem is ephemeral, so it does not outlive the session.

Two things worth knowing if you are poking at it:

- Tickers and dates are user-supplied and passed to `yfinance`. Bad values are
  handled and fail with a readable error rather than crashing the app.
- No secrets are required. `.env` files and `.streamlit/secrets.toml` are
  gitignored, and the app currently reads neither.

Fixes are handled on a best-effort basis. Please allow time for investigation
before making a report public.
