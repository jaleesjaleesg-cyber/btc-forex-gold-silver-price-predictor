"""
Risk warning engine — the SIZE of coming moves, never their direction.

Rules are fixed in results/risk_warning/PREREGISTRATION.md (written before any result was computed). In short, for each day t
(using only data up to t):

    σ̂1      one-day volatility (EWMA, λ = 0.94: the `ewma_vol` feature)
    σ̂5      typical daily move over the next 5 days (EWMA or HAR, chosen per asset on walk-forward Brier score)
    shocks  z_s = r_s / σ̂1,s−1 from days before the yearly refit date (real, fat-tailed surprises)
    S5      sums of 5 shocks drawn from that pool (filtered historical simulation)
    σ_long  standard deviation of the last 504 daily returns (the asset's recent "normal")

    P_big1  = P(|z| > 2σ_long / σ̂1)              chance of a big move tomorrow
    P_big5  = P(|S5| > 2√5σ_long / σ̂5)           chance of a big move over the next 5 days  (primary)
    P_down5 = P(S5 < −2√5σ_long / σ̂5)            chance of a big FALL over the next 5 days
    VaR/ES  97.5 % loss quantile / average loss beyond it (1 and 5 days)
    lift    = P_big5 / historical rate of 5-day big moves (climatology)
    pct     = percentile of σ̂5 among the previous 252 days

    level   EXTREME  lift ≥ 3 and pct ≥ 95 · HIGH lift ≥ 2 or pct ≥ 95 · MODERATE lift ≥ 1.25 or pct ≥ 80 · else NORMAL
    overlay HIGH UNCERTAINTY when ≥ 4 of the last 20 days broke the one-day 95 % range (about 1 expected)

Everything is causal: the truncation tests in tests/test_risk_warning.py check that no value at t changes when later rows are removed.
"""
import os
import json
import numpy as np
import pandas as pd

from config import MODELS_DIR, get_prefix
from src.volatility.models import HARLog

HORIZON = 5
Q = 0.975                       # VaR / ES level and the one-day range used by the calibration monitor
N_SIM = 50_000
SEED = 7
LONG_WIN, LONG_MIN = 504, 250   # the asset's "normal" daily move: last ~2 years
PCT_WIN = 252                   # relative level: compared with the previous year
MONITOR_WIN, MONITOR_MIN_BREACHES = 20, 4
POOL_MIN = 250
LEVELS = ('NORMAL', 'MODERATE', 'HIGH', 'EXTREME')
LEVEL_RULES = {'EXTREME': 'chance of a big 5-day move ≥ 3× usual AND risk in the top 5 % of the past year',
               'HIGH': 'chance of a big 5-day move ≥ 2× usual OR risk in the top 5 % of the past year',
               'MODERATE': 'chance ≥ 1.25× usual OR risk in the top 20 % of the past year',
               'NORMAL': 'none of the above'}


# ─────────────────────────────────────────────── building blocks (all causal)
def sigma_long(r):
    return r.rolling(LONG_WIN, min_periods=LONG_MIN).std()


def shocks(f):
    """z_s = r_s / σ̂1,s−1: how surprising each day's move was relative to the volatility known the day before."""
    z = f['log_return'] / f['ewma_vol'].shift(1)
    return z.replace([np.inf, -np.inf], np.nan)


def har_target(r, h=HORIZON):
    """ln of the root-mean-square daily return over days t+1 … t+h (NaN where the future is not complete)."""
    s = sum(r.shift(-k) ** 2 for k in range(1, h + 1)) / h
    return np.log(np.sqrt(s).clip(lower=1e-6))


def fit_har(f, cutoff):
    """HAR fitted on days whose 5-day target ends strictly before `cutoff`."""
    y = har_target(f['log_return'])
    pos = np.arange(len(f))
    end = np.minimum(pos + HORIZON, len(f) - 1)
    known = (f.index[end] < pd.Timestamp(cutoff)) & (pos + HORIZON < len(f)) & y.notna().values
    known &= f[['rv_1d', 'rv_5d', 'rv_22d']].notna().all(axis=1).values
    dates = f.index[known]
    return HARLog(with_iv=False).fit(f, dates, y.loc[dates].values)


class ShockPool:
    """The empirical shocks before a cutoff and the simulated 5-day sums (fixed seed → deterministic)."""

    def __init__(self, z, seed=SEED):
        z = np.asarray(z, float)
        z = z[np.isfinite(z)]
        if len(z) < POOL_MIN:
            raise ValueError(f"too few shocks ({len(z)}) to build the risk pool (need {POOL_MIN})")
        rng = np.random.default_rng(seed)
        self.z = np.sort(z)
        self.abs_z = np.sort(np.abs(z))
        self.s5 = np.sort(rng.choice(z, size=(N_SIM, HORIZON)).sum(axis=1))
        self.abs_s5 = np.sort(np.abs(self.s5))
        self.q_abs = float(np.quantile(self.abs_z, 0.95))     # one-day 95 % range in units of σ̂1 (calibration monitor)

    @staticmethod
    def _tail(sorted_vals, x):
        """P(value > x) for each x (vectorised)."""
        return 1.0 - np.searchsorted(sorted_vals, np.asarray(x, float), side='right') / len(sorted_vals)

    def p_abs1(self, x):
        return self._tail(self.abs_z, x)

    def p_abs5(self, x):
        return self._tail(self.abs_s5, x)

    def p_down5(self, x):
        return np.searchsorted(self.s5, -np.asarray(x, float), side='left') / len(self.s5)

    def var_es(self, horizon):
        vals = self.z if horizon == 1 else self.s5
        q = float(np.quantile(vals, 1 - Q))
        return -q, -float(vals[vals <= q].mean())                # in units of σ̂ (positive = loss)

    def to_dict(self):
        return {'z': [round(float(v), 6) for v in self.z]}

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d['z']))


def past_percentile(s, win=PCT_WIN):
    """Share of the previous `win` values that are ≤ today's value (NaN until a full year of history)."""
    v = s.values
    out = np.full(len(v), np.nan)
    for i in range(win, len(v)):
        if np.isfinite(v[i]):
            prev = v[i - win:i]
            prev = prev[np.isfinite(prev)]
            if len(prev) >= win // 2:
                out[i] = float(np.mean(prev <= v[i]))
    return pd.Series(out, index=s.index)


def events(f):
    """E1/E2/E3 for each day t (NaN where the future is not yet known) and the thresholds c1, c5 known at t."""
    r = f['log_return']
    sl = sigma_long(r)
    c1, c5 = 2 * sl, 2 * np.sqrt(HORIZON) * sl
    r1 = r.shift(-1)
    r5 = sum(r.shift(-k) for k in range(1, HORIZON + 1))
    e1 = (r1.abs() > c1).astype(float).where(r1.notna() & c1.notna())
    e2 = (r5.abs() > c5).astype(float).where(r5.notna() & c5.notna())
    e3 = (r5 < -c5).astype(float).where(r5.notna() & c5.notna())
    return pd.DataFrame({'c1': c1, 'c5': c5, 'r1': r1, 'r5': r5, 'e1': e1, 'e2': e2, 'e3': e3})


def climatology(e2):
    """Historical rate of E2 known at t: outcomes of days ≤ t−5 only (an outcome is known 5 days after its start)."""
    return e2.shift(HORIZON).expanding(min_periods=LONG_MIN).mean()


def level_of(lift, pct):
    lift, pct = np.asarray(lift, float), np.asarray(pct, float)
    pct = np.nan_to_num(pct, nan=0.0)
    return np.select([(lift >= 3.0) & (pct >= 0.95), (lift >= 2.0) | (pct >= 0.95), (lift >= 1.25) | (pct >= 0.80)],
                     ['EXTREME', 'HIGH', 'MODERATE'], default='NORMAL')


def sigma5_series(f, method, har=None):
    if method == 'EWMA':
        return f['ewma_vol']
    if method == 'Persistence':
        return f['rv_5d']
    if method == 'HAR':
        ok = f[['rv_1d', 'rv_5d', 'rv_22d']].notna().all(axis=1)
        out = pd.Series(np.nan, index=f.index)
        out[ok] = np.exp(har.predict(f, f.index[ok]))
        return out
    raise ValueError(f"unknown σ̂5 method {method!r}")


def compute(f, method, pool, har=None, sig5=None):
    """Every warning quantity for every day of `f` with ONE pool / HAR fit (callers restrict to the days that pool may serve)."""
    r = f['log_return']
    ev = events(f)
    s1 = f['ewma_vol']
    s5 = sigma5_series(f, method, har) if sig5 is None else sig5
    out = pd.DataFrame(index=f.index)
    out['sigma1'], out['sigma5'] = s1, s5
    out['p_big1'] = pool.p_abs1((ev['c1'] / s1).values)
    out['p_big5'] = pool.p_abs5((ev['c5'] / s5).values)
    out['p_down5'] = pool.p_down5((ev['c5'] / s5).values)
    bad = ~(np.isfinite(ev['c5']) & np.isfinite(s5) & (s5 > 0) & np.isfinite(s1) & (s1 > 0))
    out.loc[bad, ['p_big1', 'p_big5', 'p_down5']] = np.nan
    v1, e1_ = pool.var_es(1)
    v5, e5_ = pool.var_es(HORIZON)
    out['var1'], out['es1'] = v1 * s1, e1_ * s1
    out['var5'], out['es5'] = v5 * s5, e5_ * s5
    out['base5'] = climatology(ev['e2'])
    out['lift'] = out['p_big5'] / out['base5']
    out['pct'] = past_percentile(s5)
    out['level'] = np.where(out['lift'].notna(), level_of(out['lift'], out['pct']), None)
    breach = (r.abs() > s1.shift(1) * pool.q_abs).astype(float).where(s1.shift(1).notna())
    out['breaches20'] = breach.rolling(MONITOR_WIN, min_periods=MONITOR_WIN).sum()
    out['uncertain'] = out['breaches20'] >= MONITOR_MIN_BREACHES
    return out.join(ev)


# ─────────────────────────────────────────────── serving
def artefact_path(asset):
    return os.path.join(MODELS_DIR, f'{get_prefix(asset)}_risk_warning.json')


def load_artefact(asset):
    path = artefact_path(asset)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No risk-warning model for {asset}: run `make risk-warning` (python src/risk/evaluate_warnings.py)")
    with open(path) as fh:
        return json.load(fh)


def served_frame(asset, feats, art=None):
    """The served warning for every day of `feats` (pool and HAR from the artefact: refit date = 1 January of the latest year)."""
    art = art or load_artefact(asset)
    pool = ShockPool.from_dict(art['pool'])
    har = HARLog.from_dict(art['har']) if art['sigma5_method'] == 'HAR' else None
    return compute(feats, art['sigma5_method'], pool, har)


def risk_warning(asset, feats):
    """The warning for the last day of `feats` + the facts the dashboard needs to explain it."""
    art = load_artefact(asset)
    fr = served_frame(asset, feats, art)
    row = fr.iloc[-1]
    if not np.isfinite(row['p_big5']):
        raise ValueError(f"not enough history to compute the risk warning for {asset}")
    return {'asset': asset, 'as_of_date': str(fr.index[-1].date()), 'level': row['level'], 'uncertain': bool(row['uncertain']),
            'breaches20': int(row['breaches20']) if np.isfinite(row['breaches20']) else None,
            'p_big1': float(row['p_big1']), 'p_big5': float(row['p_big5']), 'p_down5': float(row['p_down5']),
            'base5': float(row['base5']), 'lift': float(row['lift']), 'pct': None if pd.isna(row['pct']) else float(row['pct']),
            'sigma1_pct': float(row['sigma1'] * 100), 'sigma5_pct': float(row['sigma5'] * 100),
            'big_move_threshold_5d_pct': float(row['c5'] * 100), 'big_move_threshold_1d_pct': float(row['c1'] * 100),
            'var1_pct': float(row['var1'] * 100), 'es1_pct': float(row['es1'] * 100),
            'var5_pct': float(row['var5'] * 100), 'es5_pct': float(row['es5'] * 100),
            'sigma5_method': art['sigma5_method'], 'rules': LEVEL_RULES, 'test': art.get('test', {}), 'fresh': art.get('fresh', {})}


def warning_history(asset, feats, n_days=90):
    """The served warning on each of the last `n_days` days next to what happened (5-day outcome where already known)."""
    fr = served_frame(asset, feats)
    return fr.iloc[-n_days:]
