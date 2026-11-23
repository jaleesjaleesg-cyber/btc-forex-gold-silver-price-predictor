"""
Pre-registered evaluation of the risk warning (results/risk_warning/PREREGISTRATION.md).

    python src/risk/evaluate_warnings.py

1. Walk-forward: refit every 1 January (shock pool + HAR on data before that date) and forecast that year, 2020 → end.
2. Selection (days whose 5-day outcome ends on or before VAL_END only): σ̂5 = EWMA or HAR by Brier score of P_big5 (EWMA on a tie < 0.5 %).
3. Test, read once: days after VAL_END in the frozen dataset. The fresh live days after the frozen dataset are reported separately.
4. Writes results/risk_warning/*.csv + RESULTS.md and data/models/<prefix>_risk_warning.json (the served warning).
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import ASSETS, PROCESSED_DATA_DIR, RESULTS_DIR, VAL_END, get_prefix
from src.risk.warning import (ShockPool, shocks, fit_har, compute, artefact_path, LEVELS, HORIZON, Q, LEVEL_RULES)
from src.utils.metrics import diebold_mariano

OUT = os.path.join(RESULTS_DIR, 'risk_warning')
os.makedirs(OUT, exist_ok=True)
CANDIDATES = ('EWMA', 'HAR')
BASELINE_FHS = 'Persistence'
EPISODE_FALL = {'Bitcoin': 0.25, 'Gold': 0.10, 'Silver': 0.10}
FIRST_YEAR = 2020


def load(asset, live=False):
    p = get_prefix(asset)
    path = os.path.join(PROCESSED_DATA_DIR, f"{p}_{'live_features' if live else 'features'}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col='timestamp', parse_dates=True).sort_index()


def walk_forward(f, method, last_year):
    """Out-of-sample warning frame: for each year Y the pool / HAR come from data before 1 January Y."""
    parts = []
    for y in range(FIRST_YEAR, last_year + 1):
        cut = pd.Timestamp(f'{y}-01-01')
        rows = (f.index >= cut) & (f.index < pd.Timestamp(f'{y + 1}-01-01'))
        if not rows.any():
            continue
        pool = ShockPool(shocks(f).loc[:cut - pd.Timedelta(days=1)])
        har = fit_har(f, cut) if method == 'HAR' else None
        parts.append(compute(f, method, pool, har)[rows])
    return pd.concat(parts)


def brier(p, e):
    return float(np.mean((np.asarray(p) - np.asarray(e)) ** 2))


def prob_metrics(p, e, clim):
    ok = np.isfinite(p) & np.isfinite(e) & np.isfinite(clim)
    p, e, clim = np.asarray(p)[ok], np.asarray(e)[ok], np.asarray(clim)[ok]
    b, bc = brier(p, e), brier(clim, e)
    _, dm_p = diebold_mariano(e, p, clim, h=HORIZON)
    auc = roc_auc_score(e, p) if 0 < e.sum() < len(e) else np.nan
    return {'n': int(len(e)), 'events': int(e.sum()), 'event_rate_pct': float(e.mean() * 100), 'mean_prob_pct': float(p.mean() * 100),
            'brier': b, 'brier_climatology': bc, 'brier_skill_pct': float((1 - b / bc) * 100), 'dm_p_vs_climatology': dm_p, 'auc': auc}


def alarm_metrics(alarm, e):
    ok = np.isfinite(e)
    a, e = np.asarray(alarm, bool)[ok], np.asarray(e)[ok].astype(bool)
    tp, fp, fn = int((a & e).sum()), int((a & ~e).sum()), int((~a & e).sum())
    prec = tp / max(a.sum(), 1); rec = tp / max(e.sum(), 1)
    rate_off = float(e[~a].mean()) if (~a).any() else np.nan
    return {'days': int(len(e)), 'alarm_days': int(a.sum()), 'alarm_rate_pct': float(a.mean() * 100), 'events': int(e.sum()),
            'precision_pct': prec * 100, 'recall_pct': rec * 100, 'false_alarm_rate_pct': fp / max((~e).sum(), 1) * 100,
            'f1': (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0,
            'event_rate_when_alarm_pct': prec * 100, 'event_rate_otherwise_pct': rate_off * 100 if np.isfinite(rate_off) else np.nan,
            'lift_vs_overall': (prec / e.mean()) if e.mean() > 0 else np.nan}


def reliability(p, e, edges=(0, .02, .05, .1, .2, .4, 1.0)):
    ok = np.isfinite(p) & np.isfinite(e)
    d = pd.DataFrame({'p': np.asarray(p)[ok], 'e': np.asarray(e)[ok]})
    d['bin'] = pd.cut(d['p'], list(edges), include_lowest=True)
    g = d.groupby('bin', observed=True).agg(days=('e', 'size'), mean_prob_pct=('p', 'mean'), event_rate_pct=('e', 'mean'))
    g[['mean_prob_pct', 'event_rate_pct']] *= 100
    return g.reset_index().assign(bin=lambda x: x['bin'].astype(str))


def var_backtests(fr, pool):
    """1-day VaR97.5: Kupiec (coverage) and Christoffersen (independence); ES97.5: Acerbi–Székely Z2 with a simulated p-value."""
    d = fr.dropna(subset=['r1', 'var1', 'es1'])
    x, v, es, s1 = d['r1'].values, d['var1'].values, d['es1'].values, d['sigma1'].values
    alpha = 1 - Q
    hit = (x < -v).astype(int)
    T, n1 = len(hit), int(hit.sum())
    ph = n1 / T if T else np.nan
    ll = lambda p_, k, n: (n - k) * np.log(max(1 - p_, 1e-12)) + k * np.log(max(p_, 1e-12))
    lr_uc = -2 * (ll(alpha, n1, T) - ll(ph, n1, T)) if 0 < n1 < T else (-2 * ll(alpha, n1, T) if n1 == 0 else np.nan)
    pairs = list(zip(hit[:-1], hit[1:]))
    n00 = sum(1 for a, b in pairs if a == 0 and b == 0); n01 = sum(1 for a, b in pairs if a == 0 and b == 1)
    n10 = sum(1 for a, b in pairs if a == 1 and b == 0); n11 = sum(1 for a, b in pairs if a == 1 and b == 1)
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1); p_ = (n01 + n11) / max(len(pairs), 1)
    lr_ind = -2 * (ll(p_, n01 + n11, len(pairs)) - (ll(p01, n01, n00 + n01) + ll(p11, n11, n10 + n11))) if n1 else 0.0
    z2 = float(np.sum(x * hit / (T * alpha * es)) + 1)
    rng = np.random.default_rng(11)
    sims = []
    for _ in range(2000):
        xs = s1 * rng.choice(pool.z, size=T)
        hs = xs < -v
        sims.append(np.sum(xs * hs / (T * alpha * es)) + 1)
    p_z2 = float(np.mean(np.array(sims) <= z2))
    return {'days': T, 'breaches': n1, 'expected_breaches': T * alpha, 'breach_rate_pct': ph * 100,
            'kupiec_p': float(1 - stats.chi2.cdf(lr_uc, 1)) if np.isfinite(lr_uc) else np.nan,
            'christoffersen_ind_p': float(1 - stats.chi2.cdf(lr_ind, 1)), 'es_z2': z2, 'es_z2_p': p_z2}


def episodes(f, fr, fall):
    """Falls of ≥ `fall` from a running peak; the warning level before the peak and the first HIGH-or-above warning."""
    p = f['price']
    rows, peak_i, i, n = [], 0, 0, len(p)
    while i < n:
        if p.iloc[i] >= p.iloc[peak_i]:
            peak_i = i
        if p.iloc[i] <= p.iloc[peak_i] * (1 - fall):
            pk, cross = p.index[peak_i], p.index[i]
            j = i
            while j < n - 1 and p.iloc[j + 1] < p.iloc[peak_i] * (1 - fall / 2):
                j += 1
            bottom = p.iloc[peak_i:j + 1].idxmin()
            win = fr.loc[:pk].tail(20)
            seg = fr.loc[p.index[max(peak_i - 20, 0)]:cross]
            hi = seg[seg['level'].isin(['HIGH', 'EXTREME'])]
            first = hi.index[0] if len(hi) else None
            if len(fr.loc[:pk]):
                rows.append({'peak_date': str(pk.date()), 'peak': float(p.iloc[peak_i]), 'date_fall_reached': str(cross.date()),
                             'bottom_date': str(bottom.date()), 'max_fall_pct': float((p[bottom] / p.iloc[peak_i] - 1) * 100),
                             'level_at_peak': fr['level'].get(pk), 'highest_level_20d_before_peak':
                                 (max(win['level'].dropna(), key=LEVELS.index) if win['level'].notna().any() else None),
                             'first_high_or_above': str(first.date()) if first is not None else None,
                             'price_vs_peak_at_first_high_pct': float((p[first] / p.iloc[peak_i] - 1) * 100) if first is not None else None,
                             'warned_before_peak': bool(first is not None and first <= pk)})
            i = j + 1
            peak_i = i if i < n else peak_i
            continue
        i += 1
    return rows


def main():
    rows_sel, rows_test, rows_alarm, rows_rel, rows_var, rows_epi, rows_fresh, levels_tbl = [], [], [], [], [], [], [], []
    md = ["# Risk warning — results (auto-generated by src/risk/evaluate_warnings.py)", "",
          "Rules: `PREREGISTRATION.md` (written before this script was run). Events: **E2** = 5-day move larger than 2·√5 × the asset's "
          "2-year typical daily move (primary); **E1** = 1-day move > 2× typical; **E3** = 5-day FALL beyond the E2 threshold. Probabilities "
          "by filtered historical simulation; walk-forward refit every 1 January; selection on days with outcomes complete by "
          f"{VAL_END}; test = days after {VAL_END} in the frozen dataset; fresh = live days after the frozen dataset.", ""]
    for asset in ASSETS:
        f = load(asset)
        last_year = f.index[-1].year
        fr = {m: walk_forward(f, m, last_year) for m in CANDIDATES + (BASELINE_FHS,)}
        idx = fr['EWMA'].index
        pos = f.index.get_indexer(idx)
        end = f.index[np.minimum(pos + HORIZON, len(f) - 1)]
        sel = (end <= pd.Timestamp(VAL_END)) & ((pos + HORIZON) < len(f))
        test = idx > pd.Timestamp(VAL_END)
        # ── selection
        sel_b = {}
        for m in CANDIDATES + (BASELINE_FHS,):
            d = fr[m][sel]
            met = prob_metrics(d['p_big5'], d['e2'], d['base5'])
            sel_b[m] = met['brier']
            rows_sel.append({'asset': asset, 'sigma5': m, 'role': 'baseline' if m == BASELINE_FHS else 'candidate', **met})
        clim_sel = prob_metrics(fr['EWMA'][sel]['base5'], fr['EWMA'][sel]['e2'], fr['EWMA'][sel]['base5'])
        rows_sel.append({'asset': asset, 'sigma5': 'Climatology', 'role': 'baseline', **clim_sel})
        chosen = 'HAR' if sel_b['HAR'] < sel_b['EWMA'] * (1 - 0.005) else 'EWMA'
        decision = (f"{chosen} (walk-forward Brier EWMA {sel_b['EWMA']:.5f}, HAR {sel_b['HAR']:.5f}; rule: HAR only if ≥ 0.5 % better)")
        # ── test (read once)
        ft = fr[chosen][test]
        for ev, pcol, label in (('e2', 'p_big5', 'E2 big 5-day move'), ('e1', 'p_big1', 'E1 big 1-day move'), ('e3', 'p_down5', 'E3 big 5-day fall')):
            clim = ft['e2'].copy() * np.nan
            if ev == 'e2':
                clim = ft['base5']
            else:                                                   # causal climatology of the same event
                ser = fr[chosen][ev]
                clim = ser.shift(1 if ev == 'e1' else HORIZON).expanding(min_periods=250).mean().loc[ft.index]
            for name, p in ((f'Our warning ({chosen})', ft[pcol]), ('Persistence-FHS', fr[BASELINE_FHS][test][pcol]),
                            ('Climatology', clim)):
                rows_test.append({'asset': asset, 'event': label, 'forecast': name, **prob_metrics(p, ft[ev], clim)})
        for lvl_min in ('MODERATE', 'HIGH'):
            alarm = ft['level'].isin(LEVELS[LEVELS.index(lvl_min):]).values
            for ev in ('e2', 'e3'):
                rows_alarm.append({'asset': asset, 'set': 'test', 'alarm': f'{lvl_min} or above', 'event': ev.upper(), **alarm_metrics(alarm, ft[ev].values)})
        rel = reliability(ft['p_big5'], ft['e2']); rel.insert(0, 'asset', asset); rows_rel += rel.to_dict('records')
        pool_test = ShockPool(shocks(f).loc[:pd.Timestamp(f'{last_year}-01-01') - pd.Timedelta(days=1)])
        rows_var.append({'asset': asset, 'set': 'test', **var_backtests(ft, pool_test)})
        lv = ft['level'].value_counts(normalize=True).reindex(list(LEVELS), fill_value=0) * 100
        unc = float(ft['uncertain'].mean() * 100)
        levels_tbl.append({'asset': asset, **{k: float(v) for k, v in lv.items()}, 'HIGH UNCERTAINTY overlay': unc})
        # ── fresh live days after the frozen dataset (same rules; the 2026 pool / HAR)
        fl = load(asset, live=True)
        fresh = {}
        if fl is not None and fl.index[-1] > f.index[-1]:
            pool_l = ShockPool(shocks(fl).loc[:pd.Timestamp(f'{fl.index[-1].year}-01-01') - pd.Timedelta(days=1)])
            har_l = fit_har(fl, pd.Timestamp(f'{fl.index[-1].year}-01-01')) if chosen == 'HAR' else None
            frl = compute(fl, chosen, pool_l, har_l)
            fd = frl[frl.index > f.index[-1]]
            fresh_alarm = alarm_metrics(fd['level'].isin(['HIGH', 'EXTREME']).values, fd['e2'].values)
            fresh = {'first_day': str(fd.index[0].date()), 'last_day': str(fd.index[-1].date()), **prob_metrics(fd['p_big5'], fd['e2'], fd['base5']),
                     'high_alarm_rate_pct': fresh_alarm['alarm_rate_pct'], 'high_precision_pct': fresh_alarm['precision_pct'],
                     'levels_pct': {k: float(v) for k, v in (fd['level'].value_counts(normalize=True).reindex(list(LEVELS), fill_value=0) * 100).items()}}
            rows_fresh.append({'asset': asset, **{k: v for k, v in fresh.items() if k != 'levels_pct'}, **{f'level_{k}_pct': v for k, v in fresh['levels_pct'].items()}})
            # episodes on the longest series available (live download), walk-forward refits
            fre = walk_forward(fl, chosen, fl.index[-1].year)
            for e_ in episodes(fl, fre, EPISODE_FALL[asset]):
                rows_epi.append({'asset': asset, 'fall_threshold_pct': EPISODE_FALL[asset] * 100, **e_})
        else:
            for e_ in episodes(f, fr[chosen], EPISODE_FALL[asset]):
                rows_epi.append({'asset': asset, 'fall_threshold_pct': EPISODE_FALL[asset] * 100, **e_})
        # ── served artefact: refit date = 1 January of the latest year of data
        cut = pd.Timestamp(f'{(fl if fl is not None else f).index[-1].year}-01-01')
        src = fl if fl is not None else f
        test_alarm = [r for r in rows_alarm if r['asset'] == asset and r['alarm'] == 'HIGH or above' and r['event'] == 'E2'][0]
        test_prob = [r for r in rows_test if r['asset'] == asset and r['event'].startswith('E2') and r['forecast'].startswith('Our')][0]
        art = {'asset': asset, 'sigma5_method': chosen, 'decision': decision, 'refit_date': str(cut.date()),
               'pool': ShockPool(shocks(src).loc[:cut - pd.Timedelta(days=1)]).to_dict(),
               'har': fit_har(src, cut).to_dict(), 'rules': LEVEL_RULES,
               'test': {'first_day': str(ft.index[0].date()), 'last_day': str(ft.index[-1].date()), 'days': test_alarm['days'],
                        'high_alarm_rate_pct': test_alarm['alarm_rate_pct'], 'high_precision_pct': test_alarm['precision_pct'],
                        'event_rate_otherwise_pct': test_alarm['event_rate_otherwise_pct'], 'event_rate_pct': test_prob['event_rate_pct'],
                        'brier_skill_pct': test_prob['brier_skill_pct'], 'dm_p_vs_climatology': test_prob['dm_p_vs_climatology'],
                        'auc': test_prob['auc'], 'levels_pct': {k: v for k, v in levels_tbl[-1].items() if k != 'asset'}},
               'fresh': {k: fresh[k] for k in ('first_day', 'last_day', 'n', 'brier_skill_pct', 'auc', 'high_alarm_rate_pct', 'high_precision_pct',
                                               'event_rate_pct') if k in fresh}}
        with open(artefact_path(asset), 'w') as fh:
            json.dump(art, fh, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else o)
        print(f"{asset}: σ̂5 = {decision}")

    tbl = lambda rows: pd.DataFrame(rows)
    for name, rows in (('selection.csv', rows_sel), ('test_probability.csv', rows_test), ('test_alarms.csv', rows_alarm),
                       ('test_reliability.csv', rows_rel), ('test_var_es.csv', rows_var), ('episodes.csv', rows_epi),
                       ('fresh.csv', rows_fresh), ('test_levels.csv', levels_tbl)):
        tbl(rows).to_csv(os.path.join(OUT, name), index=False)
    def fmt(df):
        """Markdown table without extra dependencies (floats to 3 decimals)."""
        if not len(df):
            return '(none)'
        cell = lambda v: (f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else ('—' if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v)))
        lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
        lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
        return "\n".join(lines)
    md += ["## Selection (walk-forward, outcomes complete by VAL_END)", "", fmt(tbl(rows_sel)[['asset', 'sigma5', 'role', 'n', 'events', 'brier', 'brier_skill_pct', 'auc']]), "",
           "## Test — probability of the events (read once)", "", fmt(tbl(rows_test)[['asset', 'event', 'forecast', 'n', 'events', 'event_rate_pct', 'mean_prob_pct', 'brier', 'brier_skill_pct', 'dm_p_vs_climatology', 'auc']]), "",
           "## Test — warning levels as alarms", "", fmt(tbl(rows_alarm)[['asset', 'alarm', 'event', 'days', 'alarm_rate_pct', 'events', 'precision_pct', 'recall_pct', 'false_alarm_rate_pct', 'f1', 'event_rate_otherwise_pct', 'lift_vs_overall']]), "",
           "## Test — share of days at each level (%)", "", fmt(tbl(levels_tbl)), "",
           "## Test — reliability of P(big 5-day move)", "", fmt(tbl(rows_rel)), "",
           "## Test — 1-day VaR97.5 / ES97.5 backtests", "", fmt(tbl(rows_var)), "",
           "## Fresh live days after the frozen dataset", "", fmt(tbl(rows_fresh)), "",
           "## Episodes — falls from a running peak (live series, walk-forward warnings)", "",
           "How to read: `first_high_or_above` = first HIGH/EXTREME day from 20 trading days before the peak until the fall threshold was reached; "
           "`price_vs_peak_at_first_high_pct` = the price on that day compared with the peak (negative BEFORE the peak = the price was still lower "
           "and rising — the warning then reflected big moves during the rally, not a forecast of the fall; negative AFTER the peak = the fall "
           "already under way). `warned_before_peak` therefore means 'risk was flagged before the top', not 'the fall was predicted'.", "",
           fmt(tbl(rows_epi)), "",
           "Deviations from the pre-registration: none in the rules. Implementation detail fixed while coding (before any result): the episode "
           "table uses the longest available series (live download) with the same yearly walk-forward refits.", ""]
    with open(os.path.join(OUT, 'RESULTS.md'), 'w') as fh:
        fh.write("\n".join(md))
    print(f"→ {OUT}/RESULTS.md and data/models/*_risk_warning.json")


if __name__ == '__main__':
    main()
