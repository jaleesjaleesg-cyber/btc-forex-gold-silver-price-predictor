"""Risk warning engine (src/risk/warning.py): no look-ahead, sensible probabilities, fixed level rules, VaR coverage, serving."""
import numpy as np
import pandas as pd
import pytest

from src.risk.warning import (ShockPool, shocks, fit_har, compute, level_of, climatology, events, risk_warning, POOL_MIN, HORIZON)
from src.risk.evaluate_warnings import var_backtests
from src.inference.prediction import load_features


def _frozen(asset):
    return load_features(asset, live=False)


@pytest.mark.parametrize('asset,method', [('Gold', 'EWMA'), ('Bitcoin', 'HAR')])
def test_warning_at_day_t_does_not_use_later_rows(asset, method):
    f = _frozen(asset)
    cut = pd.Timestamp('2025-01-01')
    pool = ShockPool(shocks(f).loc[:cut - pd.Timedelta(days=1)])
    har = fit_har(f, cut) if method == 'HAR' else None
    full = compute(f, method, pool, har)
    k = len(f) - 40                                           # drop the last 40 rows: every forecast made before them must be identical
    part = compute(f.iloc[:k], method, pool, har)
    cols = ['sigma1', 'sigma5', 'p_big1', 'p_big5', 'p_down5', 'var1', 'es1', 'var5', 'es5', 'base5', 'lift', 'pct', 'breaches20']
    a, b = full.iloc[:k][cols], part[cols]
    np.testing.assert_allclose(a.values.astype(float), b.values.astype(float), rtol=1e-12, atol=1e-15, equal_nan=True)
    assert (full.iloc[:k]['level'].fillna('-') == part['level'].fillna('-')).all()


def test_har_fit_only_uses_targets_finished_before_the_cutoff():
    f = _frozen('Gold')
    cut = pd.Timestamp('2024-01-01')
    a = fit_har(f, cut).coef
    b = fit_har(f.loc[:'2023-12-29'], cut).coef                # the last trading days of 2023 cannot have complete targets anyway
    c = fit_har(f.loc[:pd.Timestamp('2023-12-31') + pd.Timedelta(days=60)], cut).coef
    np.testing.assert_allclose(a, c, rtol=1e-12)
    assert np.all(np.isfinite(b))


def test_probability_grows_with_volatility_and_stays_in_range():
    pool = ShockPool(np.random.default_rng(0).standard_t(5, size=3000))
    x = np.array([4.0, 3.0, 2.0, 1.0, 0.5])                    # threshold / σ̂: smaller = higher volatility
    p = pool.p_abs5(x)
    assert np.all(np.diff(p) > 0) and np.all((p >= 0) & (p <= 1))
    assert np.all(pool.p_down5(x) <= p)


def test_pool_needs_enough_shocks():
    with pytest.raises(ValueError):
        ShockPool(np.ones(POOL_MIN - 1))


def test_level_rules_are_the_preregistered_ones():
    lift = np.array([0.5, 1.3, 0.9, 2.1, 1.0, 3.2, 3.2])
    pct = np.array([0.1, 0.1, 0.85, 0.1, 0.96, 0.96, 0.5])
    assert list(level_of(lift, pct)) == ['NORMAL', 'MODERATE', 'MODERATE', 'HIGH', 'HIGH', 'EXTREME', 'HIGH']


def test_climatology_uses_only_finished_outcomes():
    e = pd.Series(np.r_[np.zeros(300), np.ones(10)], index=pd.date_range('2020-01-01', periods=310))
    c = climatology(e)
    # an outcome at day s is known only from s + 5: the climatology at the first 1 (index 300) still sees only zeros
    assert c.iloc[300] == 0 and c.iloc[304] == 0 and c.iloc[305] > 0


def test_events_look_only_forward_and_thresholds_only_backward():
    f = _frozen('Silver')
    ev = events(f)
    ev_cut = events(f.iloc[:-30])
    np.testing.assert_allclose(ev['c5'].iloc[:-30], ev_cut['c5'], equal_nan=True)   # thresholds known at t
    assert ev['e2'].iloc[-HORIZON:].isna().all()                                       # no outcome without the future


def test_var_has_the_right_coverage_on_a_correct_model():
    rng = np.random.default_rng(1)
    z = rng.standard_t(6, size=20000) / np.sqrt(6 / 4)
    pool = ShockPool(z[:5000])
    n = 4000
    sig = np.full(n, 0.01)
    v1, e1 = pool.var_es(1)
    fr = pd.DataFrame({'r1': sig * z[5000:5000 + n], 'var1': v1 * sig, 'es1': e1 * sig, 'sigma1': sig})
    res = var_backtests(fr, pool)
    assert abs(res['breach_rate_pct'] - 2.5) < 0.8 and res['kupiec_p'] > 0.01


@pytest.mark.parametrize('asset', ['Bitcoin', 'Gold', 'Silver'])
def test_served_warning_is_complete(asset):
    w = risk_warning(asset, load_features(asset))
    assert w['level'] in ('NORMAL', 'MODERATE', 'HIGH', 'EXTREME')
    for k in ('p_big1', 'p_big5', 'p_down5', 'base5'):
        assert 0 <= w[k] <= 1
    assert w['p_down5'] <= w['p_big5'] and w['var5_pct'] > 0 and w['es5_pct'] >= w['var5_pct']
    assert w['test'].get('days', 0) > 100
