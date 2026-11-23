"""Complete-bar rule: the running bar of the current day is never treated as a close."""
import datetime as dt
import pandas as pd
from src.data.market_calendar import complete_bar_cutoff, drop_incomplete_bars, next_trading_day


def test_complete_bar_cutoff_crypto_is_today_utc_and_commodity_after_session():
    sunday = dt.datetime(2026, 9, 13, 10, 42, tzinfo=dt.timezone.utc)          # 06:42 ET
    assert complete_bar_cutoff('crypto', sunday) == dt.date(2026, 9, 13)
    assert complete_bar_cutoff('commodity', sunday) == dt.date(2026, 9, 13)
    monday_evening = dt.datetime(2026, 9, 14, 22, 0, tzinfo=dt.timezone.utc)   # 18:00 ET: session over
    assert complete_bar_cutoff('commodity', monday_evening) == dt.date(2026, 9, 15)
    assert complete_bar_cutoff('crypto', monday_evening) == dt.date(2026, 9, 14)  # UTC day still running


def test_drop_incomplete_bars_removes_todays_running_row():
    df = pd.DataFrame({'timestamp': pd.to_datetime(['2026-09-12', '2026-09-13']), 'price': [1.0, 2.0]})
    now = dt.datetime(2026, 9, 13, 10, 0, tzinfo=dt.timezone.utc)
    kept = drop_incomplete_bars(df, 'crypto', now)
    assert list(kept['timestamp'].dt.date) == [dt.date(2026, 9, 12)]


def test_sunday_evening_futures_bar_is_not_a_close():
    """Sunday 23:37 ET: Globex has reopened and Yahoo returns a running bar dated Sunday — it must be dropped (crypto keeps Sundays)."""
    df = pd.DataFrame({'timestamp': pd.to_datetime(['2026-11-19', '2026-11-20', '2026-11-22']), 'price': [3685.5, 3515.4, 3608.0]})
    sunday_night = dt.datetime(2026, 11, 23, 4, 37, tzinfo=dt.timezone.utc)      # 23:37 ET on Sunday 2026-11-22
    kept = drop_incomplete_bars(df, 'commodity', sunday_night)
    assert list(kept['timestamp'].dt.date) == [dt.date(2026, 11, 19), dt.date(2026, 11, 20)]
    assert list(drop_incomplete_bars(df, 'crypto', sunday_night)['timestamp'].dt.date)[-1] == dt.date(2026, 11, 22)


def test_next_trading_day_skips_weekend_for_futures_only():
    assert next_trading_day('crypto', '2026-09-11') == dt.date(2026, 9, 12)       # Saturday: crypto trades
    assert next_trading_day('commodity', '2026-09-11') == dt.date(2026, 9, 14)    # Monday
