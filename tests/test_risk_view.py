"""Risk banner, UP/DOWN signal strength, trend description and the risk track record (src/inference/risk_view.py)."""
import numpy as np
import pandas as pd
import pytest

from src.inference.risk_view import direction_strength, market_trend, risk_now, risk_track_record, WEAK_SIGNAL_BELOW
from src.inference.prediction import load_features
from src.volatility.forecast import vol_artefact


def test_direction_is_always_shown_and_a_move_smaller_than_the_noise_is_labelled_weak():
    sigma = 0.015
    small = direction_strength(0.001, sigma)                   # +0.1 % against ±1.5 % daily noise
    assert small['direction'] == 'UP' and small['strength'] == 'weak'
    assert small['signal_ratio'] == pytest.approx(0.001 / sigma)
    assert direction_strength(-0.001, sigma)['direction'] == 'DOWN'
    big = direction_strength(-WEAK_SIGNAL_BELOW * sigma * 1.01, sigma)
    assert big['direction'] == 'DOWN' and big['strength'] == 'moderate'


@pytest.mark.parametrize('bad', [(np.nan, 0.01), (0.001, 0.0), (0.001, np.nan), (0.001, -0.01)])
def test_direction_strength_refuses_invalid_input(bad):
    with pytest.raises(ValueError):
        direction_strength(*bad)


def _prices(values):
    idx = pd.date_range('2026-01-01', periods=len(values), freq='D')
    return pd.DataFrame({'price': np.asarray(values, float)}, index=idx)


def test_market_trend_labels_falling_rising_and_mixed():
    n = 120
    falling = market_trend(_prices(np.linspace(200, 100, n)))
    assert falling['label'] == 'falling' and falling['change_3m_pct'] < 0 and not falling['above_ma50']
    assert falling['from_high_pct'] < 0 and falling['from_low_pct'] == pytest.approx(0)
    assert market_trend(_prices(np.linspace(100, 200, n)))['label'] == 'rising'
    # long rise, then a sharp drop below the 50-day average: 3-month change still positive → mixed, not "rising"
    mixed = np.concatenate([np.linspace(100, 200, n - 5), [150] * 5])
    assert market_trend(_prices(mixed))['label'] == 'mixed'


def test_market_trend_uses_only_rows_up_to_as_of():
    f = _prices(np.linspace(100, 200, 150))
    day = f.index[110]
    a = market_trend(f, as_of=day)
    b = market_trend(f.loc[:day])
    assert a == b and a['as_of_date'] == str(day.date())


def test_market_trend_needs_enough_history():
    with pytest.raises(ValueError):
        market_trend(_prices(np.linspace(100, 110, 20)))


@pytest.mark.parametrize('asset', ['Bitcoin', 'Gold', 'Silver'])
def test_risk_now_regime_matches_the_training_thresholds(asset):
    r = risk_now(asset)
    v = r['volatility']
    assert v is not None, r['volatility_unavailable_reason']
    lo, hi = vol_artefact(asset)['regime_thresholds_daily_vol']
    s = v['daily_vol_pct'] / 100
    assert v['regime'] == ('low' if s <= lo else ('high' if s > hi else 'normal'))
    assert r['times_normal'] == pytest.approx(v['daily_vol_pct'] / v['train_median_daily_vol_pct'])
    assert r['trend']['label'] in ('falling', 'rising', 'mixed')


@pytest.mark.parametrize('asset', ['Bitcoin', 'Gold'])
def test_risk_track_record_uses_only_data_up_to_each_day(asset):
    feats = load_features(asset)
    full = risk_track_record(asset, 8, feats=feats)['days']
    # drop the last 3 rows: the forecasts made on the days that remain must not change
    cut = risk_track_record(asset, 5, feats=feats.iloc[:-3])['days']
    merged = cut.merge(full, on='date', suffixes=('_cut', '_full'))
    assert len(merged) == 5
    np.testing.assert_allclose(merged['forecast_pct_cut'], merged['forecast_pct_full'], rtol=1e-12)
    np.testing.assert_allclose(merged['move_pct_cut'], merged['move_pct_full'], rtol=1e-12)


def test_risk_track_record_summary_is_consistent():
    rt = risk_track_record('Gold', 30)
    d = rt['days'].dropna(subset=['forecast_pct'])
    assert rt['n_days'] == len(d) and rt['n_high'] == int((d['regime'] == 'high').sum())
    assert rt['within_forecast_pct'] == pytest.approx(np.mean(d['move_pct'].abs() <= d['forecast_pct']) * 100)
    assert len(rt['biggest']) == 5
    assert abs(rt['biggest'][0]['move_pct']) == pytest.approx(d['move_pct'].abs().max())


def test_served_prediction_keeps_up_down_and_reports_its_strength():
    from src.inference.prediction import predict_next_day
    r = predict_next_day('Gold')
    ds = r['direction_strength']
    assert r['direction'] in ('UP', 'DOWN') and ds['direction'] == r['direction']
    assert ds['signal_ratio'] == pytest.approx(abs(r['predicted_return_pct']) / r['band_sigma_pct'])
    assert ds['strength'] == ('weak' if ds['signal_ratio'] < WEAK_SIGNAL_BELOW else 'moderate')
    assert r['trend']['label'] in ('falling', 'rising', 'mixed')
