# Risk warning — pre-registration (written BEFORE any result was computed)

Written on 2026-11-23, before `src/risk/evaluate_warnings.py` existed or was run. Every rule below is fixed; any change made after
seeing a result must be listed under "Deviations" in `RESULTS.md`. (Not yet committed to git at the time of writing: the file
timestamp and the order of the session record the sequence.)

## What is forecast
The **size** of coming moves, never the direction. A warning means "a large move — up or down — is more likely than usual".

## Events (per asset, defined at day t from data ≤ t; the outcome uses only days after t)
σ_long,t = standard deviation of the last 504 daily log returns up to t (at least 250).
- **E1 (1-day big move):** |r_{t+1}| > 2·σ_long,t
- **E2 (5-day big move, PRIMARY):** |r_{t+1} + … + r_{t+5}| > 2·√5·σ_long,t
- **E3 (5-day big fall):** r_{t+1} + … + r_{t+5} < −2·√5·σ_long,t
- **Episodes (descriptive):** falls from a running peak of ≥ 10 % (Gold, Silver) / ≥ 25 % (Bitcoin).

## Forecast of the probability (filtered historical simulation)
- One-day volatility σ̂1,t = EWMA (λ = 0.94) at t (the `ewma_vol` feature).
- Shocks z_s = r_s / σ̂1,s−1, pooled from all days before the refit date (no future shocks).
- 5-day volatility σ̂5,t — candidates: **EWMA** (= σ̂1,t), **HAR** (log-HAR on rv_1d, rv_5d, rv_22d, refit every 1 January on data whose target is complete before that date).
- 5-day sum distribution S5 = 50,000 sums of 5 shocks drawn with replacement from the pool (fixed seed).
- P_big5 = P(|S5| > c5 / σ̂5), P_down5 = P(S5 < −c5 / σ̂5), P_big1 = P(|z| > c1 / σ̂1); c1 = 2σ_long, c5 = 2√5σ_long.
- VaR97.5 and ES97.5 (1 day: σ̂1 · shock quantiles; 5 days: σ̂5 · S5 quantiles).

## Selection (uses only days whose 5-day outcome ends on or before 2026-02-17 = VAL_END)
Walk-forward years 2020 … 2026-02. The σ̂5 candidate with the lowest mean Brier score of P_big5 for E2 is chosen per asset; if the
two candidates differ by less than 0.5 % in Brier score, EWMA (the simpler one) is chosen. No other tuning.

## Baselines (must be beaten)
- **Climatology:** P = expanding historical rate of E2 up to t (outcomes complete by t).
- **Persistence-FHS:** the same simulation with σ̂5 = rv_5d (the last 5 days' volatility).

## Warning levels (fixed in advance, not fitted)
lift = P_big5 / climatology; pct = percentile of σ̂5,t among σ̂5 of the previous 252 days.
- **EXTREME:** lift ≥ 3.0 and pct ≥ 95
- **HIGH:** lift ≥ 2.0 or pct ≥ 95
- **MODERATE:** lift ≥ 1.25 or pct ≥ 80
- **NORMAL:** otherwise
- **HIGH UNCERTAINTY (overlay):** ≥ 4 of the last 20 days broke the one-day 95 % range (|r_s| > σ̂1,s−1 · q0.975 of |z|); expected about 1. Reported
  next to the level, and evaluated separately.

## Test (read once)
Test = days after 2026-02-17 in the frozen dataset (to 2026-09-12). Days after the frozen dataset (live download) are reported separately as a
fresh check. Metrics: Brier score and Brier skill vs climatology (Diebold–Mariano test on Brier losses, h = 5), AUC, a reliability table, alarm
rate, precision, recall, false-alarm rate, F1 and lift for HIGH-or-above and MODERATE-or-above, the episode table (label before the peak,
first HIGH-or-above and the fall already reached at that point), the 1-day VaR97.5 Kupiec and Christoffersen tests, and the Acerbi–Székely Z2 test for ES97.5.

## Honest expectations stated in advance
- Lift for the metals, little or none for Bitcoin.
- Most HIGH days will be false alarms.
- Slow declines (like Aug–Nov 2026) will not be warned about before the peak.
