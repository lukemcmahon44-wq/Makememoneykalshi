"""Sanity checks on config validation + warnings."""

import importlib
import os


_CONFIG_ENV_KEYS = (
    "KALSHI_API_KEY_ID", "KALSHI_PRIVATE_KEY_PATH", "KALSHI_PRIVATE_KEY",
    "KALSHI_DEMO_BASE_URL", "KALSHI_PROD_BASE_URL",
    "ENVIRONMENT", "LIVE_TRADING", "SIZING_MODE", "FIXED_TRADE_SIZE_USD",
    "RECHECK_INTERVAL_HOURS", "MIN_LIQUIDITY_CONTRACTS",
)


def _reload_with(monkeypatch, env: dict):
    """Reload `config` with the given env vars, in a way pytest auto-restores."""
    for k in _CONFIG_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import config as cfg  # noqa: WPS433
    return importlib.reload(cfg)


def test_validation_flags_missing_credentials(monkeypatch):
    cfg = _reload_with(monkeypatch, {"ENVIRONMENT": "demo"})
    joined = " | ".join(cfg.validate())
    assert "KALSHI_API_KEY_ID" in joined
    assert "KALSHI_PRIVATE_KEY" in joined


def test_validation_rejects_bad_environment(monkeypatch):
    cfg = _reload_with(monkeypatch, {
        "ENVIRONMENT": "staging",
        "KALSHI_API_KEY_ID": "abc",
        "KALSHI_PRIVATE_KEY": "x",
    })
    assert any("ENVIRONMENT" in p for p in cfg.validate())


def test_validation_rejects_bad_sizing_mode(monkeypatch):
    cfg = _reload_with(monkeypatch, {
        "ENVIRONMENT": "demo",
        "KALSHI_API_KEY_ID": "abc",
        "KALSHI_PRIVATE_KEY": "x",
        "SIZING_MODE": "ALL_IN",
    })
    assert any("SIZING_MODE" in p for p in cfg.validate())


def test_validation_rejects_zero_interval(monkeypatch):
    cfg = _reload_with(monkeypatch, {
        "ENVIRONMENT": "demo",
        "KALSHI_API_KEY_ID": "abc",
        "KALSHI_PRIVATE_KEY": "x",
        "RECHECK_INTERVAL_HOURS": "0",
    })
    assert any("RECHECK_INTERVAL_HOURS" in p for p in cfg.validate())


def test_validation_passes_with_demo_setup(monkeypatch):
    cfg = _reload_with(monkeypatch, {
        "ENVIRONMENT": "demo",
        "KALSHI_API_KEY_ID": "abc",
        "KALSHI_PRIVATE_KEY": "x",
    })
    assert cfg.validate() == []


def test_warning_when_fixed_slice_too_small(monkeypatch):
    cfg = _reload_with(monkeypatch, {
        "ENVIRONMENT": "demo",
        "KALSHI_API_KEY_ID": "abc",
        "KALSHI_PRIVATE_KEY": "x",
        "SIZING_MODE": "FIXED_DOLLAR",
        "FIXED_TRADE_SIZE_USD": "0.50",
    })
    assert any("FIXED_TRADE_SIZE_USD" in w for w in cfg.warnings())


def test_warning_when_live_production(monkeypatch):
    cfg = _reload_with(monkeypatch, {
        "ENVIRONMENT": "production",
        "KALSHI_API_KEY_ID": "abc",
        "KALSHI_PRIVATE_KEY": "x",
        "LIVE_TRADING": "true",
    })
    assert any("LIVE PRODUCTION" in w for w in cfg.warnings())


def test_main_module_imports_cleanly():
    # Smoke test: importing main shouldn't execute the loop or raise.
    import main  # noqa: F401


def test_defaults_match_spec(monkeypatch):
    """Per the spec, the project must ship with these defaults out of the box."""
    cfg = _reload_with(monkeypatch, {
        "KALSHI_API_KEY_ID": "abc",
        "KALSHI_PRIVATE_KEY": "x",
    })
    assert cfg.ENVIRONMENT == "demo"
    assert cfg.LIVE_TRADING is False
    assert cfg.SIZING_MODE == "FIXED_DOLLAR"
    assert cfg.PRICE_BAND_MIN_CENTS == 95
    assert cfg.PRICE_BAND_MAX_CENTS == 99
    assert cfg.FIXED_TRADE_SIZE_USD == 1.0
    assert cfg.RECHECK_INTERVAL_HOURS == 4.0
    assert cfg.MIN_LIQUIDITY_CONTRACTS == 1
