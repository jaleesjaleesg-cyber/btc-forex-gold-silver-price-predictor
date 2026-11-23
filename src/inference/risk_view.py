"""
What the dashboard shows FIRST: how risky the next days are, what the market is doing now, and an honest direction call.

    risk_now('Gold')               risk level for the next 5 trading days + the current trend (no return model is run)
    direction_strength(r_hat, sigma)  UP / DOWN plus how strong it is: the predicted move relative to one day's typical move
    market_trend(feats)            description of the last 3 months (NOT a forecast)
    risk_track_record('Gold', 90)  the risk forecast made on each of the last days vs the move that followed (the "proof")

Why: the next-day direction is not predictable with this data (results/final_test_results.csv, walk-forward pooled,
fresh hold-out), but the SIZE of the coming moves is (results/volatility/VOLATILITY_RESULTS.md). The UP / DOWN call is
still shown, labelled with its strength: a +0.1 % lean against ±1.5 % daily noise is a WEAK signal, and the user is told so.

Nothing here is fitted or tuned: the risk forecast is the served one (src/volatility/forecast.py), the LOW / NORMAL / HIGH
thresholds were fixed on the training period (data/models/<prefix>_vol_5d.json), and WEAK_SIGNAL_BELOW is a fixed
rule of thumb, not a value chosen on any result.
"""
import numpy as np
import pandas as pd

from config import ASSET_CONFIG
from src.inference.prediction import load_features, _features_path, _window, live_data_is_current
from src.volatility.forecast import forecast_volatility_safe

# The UP / DOWN call is labelled WEAK when the predicted move is smaller than this fraction of one day's typical move (σ_t).
# Fixed rule of thumb: a lean smaller than a quarter of the daily noise cannot be told apart from noise. Not tuned on any result.
WEAK_SIGNAL_BELOW = 0.25
TREND_DAYS = 90          # calendar days described by the trend line (≈ 3 months)
TREND_MA_ROWS = 50       # moving average of the last 50 bars


def direction_strength(r_hat, sigma):
    """r_hat, sigma: decimal log return and one-day volatility at day t. Returns the UP / DOWN call and how strong it is."""
    if not (np.isfinite(r_hat) and np.isfinite(sigma) and sigma > 0):
        raise ValueError(f"direction_strength needs a finite prediction and a positive volatility (got {r_hat}, {sigma})")
    ratio = abs(r_hat) / sigma
    return {'direction': 'UP' if r_hat > 0 else 'DOWN', 'strength': 'weak' if ratio < WEAK_SIGNAL_BELOW else 'moderate',
            'signal_ratio': float(ratio), 'weak_below': WEAK_SIGNAL_BELOW, 'predicted_pct': float(r_hat * 100),
            'sigma_pct': float(sigma * 100)}


def market_trend(feats, as_of=None):
    """Plain description of the last ~3 months up to `as_of` (only rows ≤ as_of are used). Describes, does not predict."""
    p = feats['price'] if as_of is None else feats['price'].loc[:pd.Timestamp(as_of)]
    if len(p) < TREND_MA_ROWS + 1:
        raise ValueError(f"market_trend needs at least {TREND_MA_ROWS + 1} rows (got {len(p)})")
    last, t = float(p.iloc[-1]), p.index[-1]
    win = p.loc[t - pd.Timedelta(days=TREND_DAYS):]
    start = float(win.iloc[0])
    hi, hi_date, lo, lo_date = float(win.max()), win.idxmax(), float(win.min()), win.idxmin()
    ma = float(p.iloc[-TREND_MA_ROWS:].mean())
    chg = last / start - 1
    if last < ma and chg < 0:
        label = 'falling'
    elif last > ma and chg > 0:
        label = 'rising'
    else:
        label = 'mixed'
    return {'label': label, 'as_of_date': str(t.date()), 'last_close': last, 'change_3m_pct': chg * 100,
            'high_3m': hi, 'high_3m_date': str(hi_date.date()), 'from_high_pct': (last / hi - 1) * 100,
            'low_3m': lo, 'low_3m_date': str(lo_date.date()), 'from_low_pct': (last / lo - 1) * 100,
            'ma50': ma, 'above_ma50': bool(last > ma), 'window_days': TREND_DAYS}


def risk_now(asset):
    """Risk level for the next 5 trading days from the latest complete bar, plus the trend. Runs no return model."""
    feats = load_features(asset)
    Xseq, Xt, last_close, as_of, _ = _window(asset, feats)
    live = _features_path(asset).endswith('_live_features.csv')
    vol, err = forecast_volatility_safe(asset, feats, as_of, live, (Xseq, Xt))
    out = {'asset': asset, 'as_of_date': str(as_of.date()), 'last_close': last_close, 'volatility': vol,
           'volatility_unavailable_reason': err, 'trend': market_trend(feats),
           'data_source': 'live download' if live else 'stored dataset', 'data_is_current': live_data_is_current(asset)}
    if vol:
        out['times_normal'] = vol['daily_vol_pct'] / vol['train_median_daily_vol_pct']
    return out


def risk_track_record(asset, n_days=90, feats=None):
    """
    The served risk forecast re-made on each of the last `n_days` days t (seeing only rows ≤ t) next to the move of day t+1.
    Returns {'days': DataFrame, summary numbers}. The forecast is the expected typical (RMS) daily move over the next 5
    trading days, so on a single day the actual move is often smaller and sometimes several times larger.
    """
    if feats is None:
        feats = load_features(asset)
    live = _features_path(asset).endswith('_live_features.csv')
    idx = feats.index
    n_days = int(min(n_days, len(feats) - 31))
    rows = []
    for i in range(len(feats) - 1 - n_days, len(feats) - 1):
        t = idx[i]
        hist = feats.iloc[:i + 1]                                   # nothing after day t
        Xseq, Xt, _, _, _ = _window(asset, hist)
        vol, err = forecast_volatility_safe(asset, hist, t, live, (Xseq, Xt))
        move = float(np.log(feats['price'].iloc[i + 1] / feats['price'].iloc[i]))
        rows.append({'date': t, 'target_date': idx[i + 1], 'forecast_pct': vol['daily_vol_pct'] if vol else np.nan,
                     'regime': vol['regime'] if vol else None, 'move_pct': move * 100,
                     'normal_pct': vol['train_median_daily_vol_pct'] if vol else np.nan})
    d = pd.DataFrame(rows)
    ok = d['forecast_pct'].notna()
    s = d[ok]
    if s.empty:
        return {'days': d, 'n_days': 0}
    absm = s['move_pct'].abs()
    big = s.loc[absm.sort_values(ascending=False).index[:5]]
    hi = s['regime'] == 'high'
    return {'days': d, 'n_days': int(len(s)), 'first_day': str(s['date'].iloc[0].date()), 'last_target': str(s['target_date'].iloc[-1].date()),
            'mean_forecast_pct': float(s['forecast_pct'].mean()), 'actual_rms_move_pct': float(np.sqrt(np.mean(s['move_pct'] ** 2))),
            'normal_pct': float(s['normal_pct'].iloc[-1]), 'within_forecast_pct': float(np.mean(absm <= s['forecast_pct']) * 100),
            'n_high': int(hi.sum()), 'mean_abs_move_high': float(absm[hi].mean()) if hi.any() else np.nan,
            'mean_abs_move_other': float(absm[~hi].mean()) if (~hi).any() else np.nan,
            'max_move_vs_forecast': float((absm / s['forecast_pct']).max()),
            'biggest': [{'date': str(r.date.date()), 'target_date': str(r.target_date.date()), 'move_pct': r.move_pct,
                         'forecast_pct': r.forecast_pct, 'regime': r.regime} for r in big.itertuples()],
            'data_source': 'live download' if live else 'stored dataset'}
