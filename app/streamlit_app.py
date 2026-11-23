"""
Streamlit dashboard — next-day price forecasting for Bitcoin, Gold and Silver.

Every number shown is read from the project's own artefacts (data/processed, data/models, results/) or
produced live by src/inference/prediction.py. Nothing is hard-coded.
The user never chooses a model: every trained model runs automatically and their average is shown ("Refresh forecast" reruns it).

Layout — a guided page: each section answers one question, details sit in expanders
  1 Today                   ① market snapshot (3 numbers) · ② tomorrow's forecast (price, UP/DOWN, likely range, signal strength)
                            · ③ market risk (risk level from src/inference/risk_view.py + "did the warning work?") · ④ how much to trust it
                            (3 numbers + verdict; full table and per-model check in expanders) · ⑤ price chart (forecast + range, optional
                            indicators) · where to go next
  2 Test a Past Day         pick a day the models never saw, forecast it, reveal the real price and the error
  3 How Accurate Is It?     1 short answer · 2 all three assets · 3 risk forecast · 4 likely range · 5 full details for examiners (expanders)
  4 How It Works            target, data, split, inputs, models, the average, the range, the risk forecast, metrics, limitations
All visible text is written in plain English (technical terms are kept but explained); numbers and calculations come from the project code.
"""
import os
import sys
import time
import logging
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import (PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR, PREDICTION_HORIZON_DAYS,
                    TRAIN_END, VAL_END, ASSETS, ASSET_CONFIG, get_prefix, load_model_status)
from src.inference.prediction import (predict_next_day, predict_for_date, prediction_history, combined_members, live_data_is_current,
                                      recent_track_record,
                                      COMBINED, DISCLAIMER)
from src.inference.risk_view import risk_now, risk_track_record, WEAK_SIGNAL_BELOW
from src.data.market_calendar import expected_last_complete_bar
from src.models.registry import ROLLING_BASELINES
from src.data.sync_live_data import update_live_data

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Next-Day Price Forecast — Bitcoin, Gold, Silver", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  .stApp { background-color: #0E1117; }
  h1, h2, h3 { font-family: 'Inter', sans-serif; color: #FFFFFF; font-weight: 700; letter-spacing: -0.3px; }
  h1 { font-size: 1.9rem !important; margin-bottom: 0.2rem !important; }
  h2 { font-size: 1.35rem !important; margin-top: 1.2rem !important; }
  h3 { font-size: 1.05rem !important; }
  div[data-testid="stMetricValue"] { font-size: 1.7rem; }
  div[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #A0A4AB; }
  .overview { background: #161B24; border: 1px solid #262C38; border-radius: 8px; padding: 0.8rem 1.1rem; margin-bottom: 0.6rem;
              font-size: 0.93rem; line-height: 1.6; color: #D6D9DE; }
  .overview b { color: #FFFFFF; }
  .final { background: linear-gradient(135deg, #12202A 0%, #161B24 100%); border: 1px solid #1F6F66; border-radius: 10px;
           padding: 1rem 1.3rem; margin: 0.4rem 0 0.8rem 0; }
  .final .lbl { color: #A0A4AB; font-size: 0.85rem; }
  .final .big { font-size: 2.1rem; font-weight: 700; color: #FFFFFF; font-family: 'Inter', sans-serif; }
  .final .up { color: #2ECC71; } .final .down { color: #FF5252; }
  .live { background: #161B24; border: 1px solid #2E3A4D; border-radius: 8px; padding: 0.55rem 0.9rem; margin: 0.3rem 0 0.7rem 0;
          color: #D6D9DE; font-size: 0.9rem; line-height: 1.55; }
  .live b { color: #FFFFFF; } .live .dot { color: #00E0C6; font-weight: 700; }
  .intro { color: #B8BCC4; font-size: 0.98rem; line-height: 1.6; margin: -0.2rem 0 0.9rem 0; max-width: 60rem; }
  .step { margin: 2.2rem 0 0.7rem 0; padding-top: 1.1rem; border-top: 1px solid #262C38; }
  .step .num { display: inline-block; width: 1.75rem; height: 1.75rem; line-height: 1.75rem; text-align: center; border-radius: 50%;
               background: #1F2A3A; color: #00E0C6; font-weight: 700; font-size: 0.95rem; margin-right: 0.6rem; vertical-align: middle; }
  .step .ttl { font-family: 'Inter', sans-serif; font-size: 1.3rem; font-weight: 700; color: #FFFFFF; vertical-align: middle; }
  .step .q { color: #A0A4AB; font-size: 0.92rem; margin: 0.25rem 0 0 2.35rem; }
  .verdict { background: #161B24; border-left: 3px solid #00E0C6; padding: 0.6rem 0.9rem; margin: 0.6rem 0; color: #D6D9DE;
             font-size: 0.95rem; line-height: 1.55; border-radius: 0 6px 6px 0; }
  .verdict b { color: #FFFFFF; }
  .next { background: #161B24; border: 1px solid #262C38; border-radius: 8px; padding: 0.8rem 1.1rem; margin: 2rem 0 0.5rem 0;
          color: #D6D9DE; font-size: 0.93rem; line-height: 1.8; }
  .next b { color: #FFFFFF; }
  .risk { border-radius: 10px; padding: 0.85rem 1.2rem; margin: 0.3rem 0 0.8rem 0; line-height: 1.6; font-size: 0.95rem; color: #E6E8EB; }
  .risk .head { font-size: 1.45rem; font-weight: 700; font-family: 'Inter', sans-serif; }
  .risk .sub { color: #A0A4AB; font-size: 0.8rem; }
  .risk-high { background: #2A1215; border: 1px solid #FF5252; } .risk-high .head { color: #FF6B6B; }
  .risk-normal { background: #2A2412; border: 1px solid #F1C40F; } .risk-normal .head { color: #F5D547; }
  .risk-low { background: #0F2A1E; border: 1px solid #2ECC71; } .risk-low .head { color: #2ECC71; }
  .risk-na { background: #161B24; border: 1px solid #262C38; } .risk-na .head { color: #D6D9DE; }
  .footer { color: #8A8F98; font-size: 0.8rem; margin-top: 2rem; border-top: 1px solid #262C38; padding-top: 0.6rem; }
</style>""", unsafe_allow_html=True)

COLOR = {"Bitcoin": "#F7931A", "Gold": "#FFD700", "Silver": "#C0C0C0"}
ACCENT = "#00E0C6"          # our forecast / model colour
NAIVE = "#9AA0A6"           # "tomorrow = today" and reference lines
BAND = "rgba(255,255,255,0.35)"
# Colours carry a meaning and are used for nothing else: green / red = the price went up / down, red = high risk.
UP_COL, DOWN_COL, RISK_HIGH = "#2ECC71", "#FF5252", "#FF5252"
SPLIT_VAL, SPLIT_TEST = "#8E7CC3", "#4E7FA8"   # neutral shading for the validation and test periods (no red / green)


# ─────────────────────────────────────────────── data loaders (all from project artefacts)
@st.cache_data(ttl=300)
def load_features(asset: str) -> Optional[pd.DataFrame]:
    p = get_prefix(asset)
    live = os.path.join(PROCESSED_DATA_DIR, f'{p}_live_features.csv')
    path = live if os.path.exists(live) else os.path.join(PROCESSED_DATA_DIR, f'{p}_features.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['timestamp']).sort_values('timestamp')
    df.attrs['source'] = 'live download' if path == live else 'frozen dataset'
    return df


@st.cache_data(ttl=300)
def load_csv(name: str) -> Optional[pd.DataFrame]:
    path = os.path.join(RESULTS_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data(ttl=300)
def load_test_predictions(asset: str) -> Optional[pd.DataFrame]:
    return prediction_history(asset)


def _data_stamp(asset: str) -> float:
    """Modification time of the feature file in use, so cached risk numbers refresh after a live download."""
    p = get_prefix(asset)
    live = os.path.join(PROCESSED_DATA_DIR, f'{p}_live_features.csv')
    path = live if os.path.exists(live) else os.path.join(PROCESSED_DATA_DIR, f'{p}_features.csv')
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


@st.cache_data(ttl=300)
def cached_risk_now(asset: str, stamp: float):
    return risk_now(asset)


@st.cache_data(ttl=600)
def cached_risk_track(asset: str, stamp: float, n_days: int):
    return risk_track_record(asset, n_days)


@st.cache_data(ttl=600, show_spinner=False)
def cached_forecast(asset: str, stamp: float):
    """The served next-day forecast (all models + their average); recomputed when new prices arrive or on Refresh."""
    return predict_next_day(asset, COMBINED)


def section(num, title, question):
    """Numbered section heading of the guided page: one question per section."""
    st.markdown(f"<div class='step'><span class='num'>{num}</span><span class='ttl'>{title}</span>"
                f"<div class='q'>{question}</div></div>", unsafe_allow_html=True)


def test_row(te: Optional[pd.DataFrame], asset: str, model: str) -> Optional[pd.Series]:
    """One model's single test-set evaluation (results/final_test_results.csv)."""
    if te is None:
        return None
    r = te[(te['asset'] == asset) & (te['model'] == model)]
    return r.iloc[0] if len(r) else None


def layout(fig, height, title=None, legend=True):
    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=0, r=0, t=48 if title else 24, b=0), showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=11)))
    if fig.layout.hovermode is None:
        fig.update_layout(hovermode="x unified")
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15)))
    return fig


def pct(x, d=1):
    return "—" if x is None or pd.isna(x) else f"{x:.{d}f}%"


def usd(x):
    return "—" if x is None or pd.isna(x) else f"${x:,.2f}"


def usd_md(x):
    """Dollar amount for st.markdown / st.caption, where a bare '$' would start LaTeX."""
    return usd(x).replace('$', '\\$')


SOURCE_TXT = {'live download': 'latest downloaded prices', 'latest live download': 'latest downloaded prices',
              'stored dataset': 'stored dataset', 'frozen dataset': 'stored dataset',
              'frozen dataset (stored test period)': 'stored dataset (test period)'}


def src_txt(s):
    """Plain-English name of a data source label produced by the inference code."""
    return SOURCE_TXT.get(s, s)


def move_txt(p):
    """+3.2 → 'up 3%', −3.2 → 'down 3%'."""
    return f"{'up' if p >= 0 else 'down'} {abs(p):.0f}%"


TAB_LABELS = ["Today", "Test a Past Day", "How Accurate Is It?", "How It Works"]

# ─────────────────────────────────────────────── sidebar: asset + refresh
st.sidebar.title("Price Forecast")
_qp = st.query_params                                    # deep links for demos: ?asset=Gold
_default_asset = ASSETS.index(_qp.get('asset')) if _qp.get('asset') in ASSETS else 0
asset = st.sidebar.selectbox("Choose an asset", ASSETS, index=_default_asset)
status_all = load_model_status()
status = status_all.get(asset, {})
members = combined_members(asset)
cv_best = status.get('primary_model')
if st.sidebar.button("Refresh forecast", type="primary", width='stretch',
                     help="Checks Yahoo Finance for a newer price, then runs all the models again.",
                     on_click=lambda: st.session_state.update(open_tab=TAB_LABELS[0])):
    if not live_data_is_current(asset):
        with st.spinner("Checking Yahoo Finance for a newer price…"):
            _ok = update_live_data(asset, refresh_external=True)
        st.session_state[f'sync_failed_{asset}'] = not _ok
    st.cache_data.clear()
    st.session_state.pop('run_score', None)                 # the per-model check is recomputed after a refresh
if st.sidebar.button("Get latest prices", width='stretch',
                     help="Downloads the newest prices from Yahoo Finance. The stored dataset behind the reported results is never changed."):
    with st.spinner("Downloading the latest prices from Yahoo Finance…"):
        ok = update_live_data(asset, refresh_external=True)
    st.session_state[f'sync_failed_{asset}'] = not ok
    if ok:
        st.cache_data.clear(); st.sidebar.success("Latest prices downloaded.")
    else:
        st.sidebar.warning("Yahoo Finance did not send all the data we need (it may be busy or offline), so nothing was changed: "
                           "the app keeps using the previous prices. Please try again in a minute.")

# Keep the live data current without a button press: when a newer daily bar should exist, download it. Checked at most
# once every 30 minutes per asset (and never more often while Yahoo has not yet published the bar).
_live_path = os.path.join(PROCESSED_DATA_DIR, f'{get_prefix(asset)}_live_features.csv')
_checked = st.session_state.get(f'live_check_{asset}', 0)
_fresh = os.path.exists(_live_path) and (time.time() - os.path.getmtime(_live_path)) < 30 * 60
if not (live_data_is_current(asset) or _fresh or time.time() - _checked < 30 * 60):
    st.session_state[f'live_check_{asset}'] = time.time()
    with st.spinner(f"Checking Yahoo Finance for new {asset} prices…"):
        _ok = update_live_data(asset, refresh_external=True)
        st.session_state[f'sync_failed_{asset}'] = not _ok     # False → offline / rate-limited (warning shown under the title)
        if _ok:
            st.cache_data.clear()

df = load_features(asset)
te_all = load_csv('final_test_results.csv')
cv_all = load_csv('cv_results.csv')
hist = load_test_predictions(asset)
comb_test = test_row(te_all, asset, COMBINED)
naive_test = test_row(te_all, asset, 'Naive')
tm = status.get('combined_test') or {}; tn = status.get('test_naive') or {}

st.sidebar.markdown("---")
st.sidebar.caption(f"**Models:** average of {len(members)} ({', '.join(members) or 'none'})  \n"
                   f"**Forecast for:** the next {PREDICTION_HORIZON_DAYS} trading day  \n"
                   f"**Data:** Yahoo Finance · {src_txt(df.attrs.get('source', '—')) if df is not None else '—'} · latest price "
                   f"{df['timestamp'].iloc[-1].date() if df is not None else '—'}")

# ─────────────────────────────────────────────── shared results for the guided page
TREND_TXT = {'falling': 'Falling', 'rising': 'Rising', 'mixed': 'No clear trend'}
try:
    rk = cached_risk_now(asset, _data_stamp(asset))
except Exception as e:                                   # e.g. no processed data yet: the page explains what to run
    rk = None
    logger.warning(f"risk summary unavailable for {asset}: {e}")
fc, fc_err = None, None
if df is not None and members:
    try:
        with st.spinner(f"Running {len(members)} models…"):
            fc = cached_forecast(asset, _data_stamp(asset))
    except Exception as e:
        fc_err = str(e)
        logger.exception("Dashboard prediction error")

# ─────────────────────────────────────────────── header
st.title(f"{asset} — Next-Day Price Forecast")
st.markdown(f"<div class='intro'>This page answers five questions about {asset}, one after another: what is happening now, what our "
            "models expect for the next trading day, how risky the market is, how much you can trust the forecast, and what the price "
            "has done. Scroll down step by step.</div>", unsafe_allow_html=True)
if df is not None and not live_data_is_current(asset):
    expected_bar = expected_last_complete_bar(ASSET_CONFIG[asset]['type'])
    if st.session_state.get(f'sync_failed_{asset}'):
        st.warning(f"Could not download new prices from Yahoo Finance (no internet, or Yahoo is busy). This page uses prices up to "
                   f"**{df['timestamp'].iloc[-1].date()}**, not the latest market close ({expected_bar}). Press **Get latest prices** to try again.")
    else:
        delay = ("Bitcoin's daily price usually appears a few hours after midnight UTC" if ASSET_CONFIG[asset]['type'] == 'crypto'
                 else "gold and silver prices appear after 5 pm New York time, and there is no price on market holidays")
        st.info(f"Yahoo Finance has not published the price for {expected_bar} yet ({delay}). This page uses the latest price available, "
                f"from {df['timestamp'].iloc[-1].date()}.")

# a button press reruns the script; `open_tab` (set by the button's on_click) keeps the user on the tab they pressed it in
_open = st.session_state.get('open_tab', TAB_LABELS[0])
tab_fc, tab_demo, tab_perf, tab_meth = st.tabs(TAB_LABELS, default=_open if _open in TAB_LABELS else TAB_LABELS[0])

# ═══════════════════════════════════════════════ 1. TODAY — a guided page, one question per section
with tab_fc:
    if df is None or len(df) < 2:
        st.info("No price data found yet. Run `make preprocess` first.")
    else:
        latest_date = df['timestamp'].iloc[-1].strftime("%Y-%m-%d")
        latest_price, prev_price = df['price'].iloc[-1], df['price'].iloc[-2]
        tr = rk['trend'] if rk else None

        # ── ① market snapshot
        section(1, "Market snapshot", f"What is happening to {asset} right now?")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Latest price ({latest_date})", usd(latest_price),
                  f"{(latest_price / prev_price - 1) * 100:+.2f}% vs the day before")
        c2.metric("Trend over 3 months", f"{TREND_TXT[tr['label']]}, {move_txt(tr['change_3m_pct'])}" if tr else "—",
                  help="What has happened in the last 3 months — a description, not a prediction.")
        c3.metric("Typical daily move (last 30 days)", f"±{df['log_return'].tail(30).std() * 100:.2f}%",
                  help="How much the price usually moved in one day over the last 30 days (also called volatility).")
        with st.expander("More price details"):
            d1, d2, d3 = st.columns(3)
            d1.metric("Highest in the last 30 days", usd(df['price'].tail(30).max()))
            d2.metric("Lowest in the last 30 days", usd(df['price'].tail(30).min()))
            if tr:
                d3.metric("3-month high", usd(tr['high_3m']), f"{tr['from_high_pct']:+.1f}% from it", delta_color="off")
                st.caption(f"The price is {'above' if tr['above_ma50'] else 'below'} its 50-day average ({usd(tr['ma50'])}).")
            if df.attrs.get('source') == 'live download' and ASSET_CONFIG[asset]['type'] == 'commodity':
                st.caption("Gold and silver prices follow the futures contract that is traded most. When Yahoo switches to the next contract, "
                           "recent prices can differ slightly from our stored data.")

        # ── ② tomorrow's forecast
        section(2, "Tomorrow's forecast", "What do our models expect for the next trading day?")
        if fc is None:
            st.error(f"Sorry, the forecast could not be made: {fc_err or 'no trained models found'}")
        else:
            r = fc
            target, up = r['target_date'], r['direction'] == 'UP'
            band = r.get('uncertainty_band')
            st.markdown(
                f"<div class='final'><div class='lbl'>Forecast for {target} · average of {len(r['individual'])} models · "
                f"latest price {usd(r['current_price'])} ({r['as_of_date']})</div>"
                f"<span class='big'>{usd(r['predicted_price'])}</span> &nbsp; "
                f"<span class='big {'up' if up else 'down'}'>{'UP ▲' if up else 'DOWN ▼'}</span> &nbsp; "
                f"<span style='color:#D6D9DE;font-size:1.1rem'>{r['predicted_return_pct']:+.2f}% "
                f"({r['predicted_price'] - r['current_price']:+,.2f} USD)</span>"
                + (f"<div class='lbl' style='margin-top:0.35rem'>Likely range: <b style='color:#FFFFFF'>{usd(band[0])} – {usd(band[1])}</b> "
                   f"· the price lands inside it on about 2 out of 3 days</div>" if band else "")
                + "</div>", unsafe_allow_html=True)
            ds = r.get('direction_strength') or {}
            if ds.get('strength') == 'weak':
                st.caption(f"**Signal: WEAK.** The models expect {r['predicted_return_pct']:+.2f}%, but a normal day moves ±{ds['sigma_pct']:.2f}%. "
                           f"That is too small to trust — {r['direction']} is only a small hint. See ④ below.")
            elif ds:
                st.caption(f"**Signal: MEDIUM.** The models expect {r['predicted_return_pct']:+.2f}% — bigger than usual for them. "
                           "But they have not been good at guessing the direction before. See ④ below.")
            # what each model predicts (the forecast above is their plain average)
            if r.get('individual'):
                st.markdown(f"**What each model predicts for {target}**")
                ind = pd.DataFrame(r['individual'])
                ind_tbl = pd.DataFrame({'Model': ind['model'],
                                        'Predicted price': ind['predicted_price'].map(usd),
                                        'Change': ind['predicted_return_pct'].map(lambda v_: f"{v_:+.2f}%"),
                                        'Direction': ind['direction'].map(lambda d_: 'UP ▲' if d_ == 'UP' else 'DOWN ▼')})
                ind_tbl = pd.concat([ind_tbl, pd.DataFrame([{'Model': 'Average of all models (our forecast)',
                                                             'Predicted price': usd(r['predicted_price']),
                                                             'Change': f"{r['predicted_return_pct']:+.2f}%",
                                                             'Direction': 'UP ▲' if up else 'DOWN ▼'}])], ignore_index=True)
                st.dataframe(ind_tbl, width='stretch', hide_index=True)
                n_up_ = int((ind['direction'] == 'UP').sum())
                st.caption(f"{n_up_} of {len(ind)} models say UP. The forecast above is the plain average of these {len(ind)} models. "
                           "To see how each model did on recent days, open \"What does each model say — and how did it do lately?\" in ④.")

        # ── ③ market risk
        section(3, "Market risk", "Is anything unusual happening? How big could the next moves be?")
        v = rk['volatility'] if rk else None
        if rk is None:
            st.info("The risk level is not available right now.")
        elif v is None:
            st.markdown(f"<div class='risk risk-na'><div class='head'>Risk level not available</div>Some data is missing. "
                        f"Press <b>Get latest prices</b>.<div class='sub'>Details: {rk['volatility_unavailable_reason'] or 'no volatility model'}</div></div>",
                        unsafe_allow_html=True)
        else:
            reg = v['regime']
            head = {'high': "⚠ HIGH RISK", 'normal': "NORMAL RISK", 'low': "LOW RISK"}[reg]
            meaning = {'high': f"{asset}'s price may jump a lot in the next few days. It could go up or down — we don't know which.",
                       'normal': f"{asset}'s price should move a normal amount in the next few days. Up or down — we don't know which.",
                       'low': f"{asset}'s price should be calmer than usual in the next few days. Up or down — we don't know which."}[reg]
            st.markdown(
                f"<div class='risk risk-{reg}'><div class='head'>{head}</div>{meaning}<br>"
                f"Normal day: ±{v['train_median_daily_vol_pct']:.2f}%. Expected now: <b>±{v['daily_vol_pct']:.2f}% a day</b> "
                f"({rk['times_normal']:.1f}× {'bigger' if rk['times_normal'] >= 1 else 'the normal size'}), "
                f"about ±{v['five_day_vol_pct']:.1f}% over the next {v['horizon_days']} trading days."
                f"<div class='sub'>Based on the price of {rk['as_of_date']}. Risk levels compare the forecast with {asset}'s normal price swings "
                f"in 2018–2025.</div></div>", unsafe_allow_html=True)
            t_ = v['test']
            method_txt = {'Ridge': 'Ridge model', 'HAR-IV': 'HAR model + gold fear index (GVZ)', 'HAR': 'HAR model',
                          'EWMA': 'average of recent price swings (EWMA)', 'LightGBM': 'LightGBM', 'CatBoost': 'CatBoost',
                          'Persistence': "copy of the last 5 days"}.get(v['method'], v['method'])
            vt_ = t_.get('RMSE_vs_ref_pct_vs_persistence')
            if v['is_ml'] and vt_ is not None:
                st.caption(f"Made by the **{method_txt}**. In tests on {t_['n']} new days it was **{abs(vt_):.0f}% better** than just copying "
                           "the last 5 days. We can't guess up or down, but we can guess how big the moves will be.")
            else:
                st.caption(f"Made by a simple rule: **{method_txt}**. No model did better for {asset}."
                           + (f" In tests it was {abs(vt_):.1f}% {'better' if vt_ < 0 else 'worse'} than copying the last 5 days." if vt_ is not None else ""))
        try:
            rt = cached_risk_track(asset, _data_stamp(asset), 90)
        except Exception as e:
            rt = None
            st.caption(f"The risk warning record is not available right now ({e}).")
        if rt and rt.get('n_days'):
            n_big_high = sum(b['regime'] == 'high' for b in rt['biggest'])
            st.markdown(f"**Did the warning work?** In the last {rt['n_days']} days, **{n_big_high} of the 5 biggest moves** happened on days "
                        "marked HIGH RISK.")
            with st.expander("See the risk record — last 90 days"):
                dd = rt['days'].dropna(subset=['forecast_pct'])
                frk = go.Figure()
                hi = (dd['regime'] == 'high').values                 # shade the days the forecast (made the day before) said HIGH
                runs, start = [], None
                for j, h in enumerate(hi):
                    if h and start is None:
                        start = j
                    if (not h or j == len(hi) - 1) and start is not None:
                        end = j if h else j - 1
                        runs.append((dd['target_date'].iloc[start], dd['target_date'].iloc[end]))
                        start = None
                for x0, x1 in runs:
                    frk.add_vrect(x0=x0 - pd.Timedelta(hours=12), x1=x1 + pd.Timedelta(hours=12), fillcolor=RISK_HIGH, opacity=0.08, line_width=0)
                frk.add_trace(go.Bar(x=dd['target_date'], y=dd['move_pct'], name='real daily move (up = green, down = red)',
                                     marker_color=np.where(dd['move_pct'] >= 0, UP_COL, DOWN_COL)))
                for sgn in (1, -1):
                    frk.add_trace(go.Scatter(x=dd['target_date'], y=sgn * dd['forecast_pct'], mode='lines', name='risk forecast (± expected move)',
                                             line=dict(color=ACCENT, width=2), showlegend=sgn == 1))
                    frk.add_trace(go.Scatter(x=dd['target_date'], y=sgn * dd['normal_pct'], mode='lines', name=f'normal for {asset}',
                                             line=dict(color=NAIVE, width=1, dash='dot'), showlegend=sgn == 1))
                frk.update_yaxes(ticksuffix='%', title_text='daily price move (%)')
                frk.update_xaxes(title_text='date')
                st.plotly_chart(layout(frk, 340, "Real daily moves vs the risk forecast made the day before"), use_container_width=True)
                st.caption("**What this shows:** bars are the real daily moves; the blue lines are how big we expected them to be; "
                           "light red shading marks days we called HIGH RISK. **What to take from it:** when the lines widen, the bars usually "
                           "get bigger too — the warning tracks the size of moves, not their direction.")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Expected daily move (average)", f"±{rt['mean_forecast_pct']:.2f}%")
                k2.metric("Real typical daily move", f"±{rt['actual_rms_move_pct']:.2f}%", help="The typical size of the real daily moves (root-mean-square).")
                k3.metric("Days within the expected move", f"{rt['within_forecast_pct']:.0f}%",
                          help="About 68% means the forecast had the right size.")
                k4.metric("Days marked HIGH RISK", f"{rt['n_high']} of {rt['n_days']}")
                big_txt = '; '.join(f"{b['target_date']} **{b['move_pct']:+.1f}%** (marked {b['regime'].upper()})" for b in rt['biggest'])
                st.markdown(f"**Biggest 5 moves:** {big_txt}.")
                st.caption(f"Each forecast used only older data. It cannot tell up or down, or the exact day — single shock days were up to "
                           f"{rt['max_move_vs_forecast']:.1f}× the expected move.")

        # ── ④ how much to trust it
        section(4, "How much can you trust the forecast?", "How did the same method do on days the models never saw?")
        if tm and tn and comb_test is not None and naive_test is not None:
            beats = comb_test['DM_pvalue'] < 0.05 and comb_test['RMSE_ret'] < naive_test['RMSE_ret']
            cov = (status.get('band') or {}).get('test_conditional_coverage_pct')
            q1, q2, q3 = st.columns(3)
            q1.metric("Direction right", f"{tm['DirAcc_pct']:.0f}%",
                      help=f"On {tm['n_test']} test days. A coin flip gets 50%. p = {tm['DirAcc_pvalue']:.2f} (below 0.05 would mean real skill).")
            q2.metric('Better than "tomorrow = today"?', "Yes" if beats else "No",
                      help=f"Statistical test (Diebold–Mariano) p = {comb_test['DM_pvalue']:.2f}; a real difference needs p below 0.05.")
            q3.metric("Price inside the likely range", f"{cov:.0f}%" if cov is not None else "—", help="Target: 68% (about 2 out of 3 days).")
            st.markdown(f"<div class='verdict'><b>In short:</b> the UP/DOWN guess is about as good as flipping a coin, and the price guess is "
                        f"{'better than' if beats else 'no better than'} simply saying \"tomorrow's price = today's price\". "
                        "The likely range and the risk level <b>are</b> useful: they tell you how much the price could move.</div>",
                        unsafe_allow_html=True)
            if pd.notna(comb_test.get('UpCalls_pct')) and comb_test['UpCalls_pct'] >= 80:
                st.caption(f"Why the direction is weak for {asset}: the models said UP on {comb_test['UpCalls_pct']:.0f}% of test days, because "
                           f"{asset} mostly went up in the past.")
            with st.expander(f"Full comparison — {int(comb_test['n_test'])} test days ({status['test_period'][0]} → {status['test_period'][1]})"):
                rel_tbl = pd.DataFrame({
                    '': ['Our forecast', 'Simple guess (tomorrow = today)'],
                    'Average error (MAE)': [usd(comb_test['MAE_usd']), usd(naive_test['MAE_usd'])],
                    'Typical error (RMSE)': [usd(comb_test['RMSE_usd']), usd(naive_test['RMSE_usd'])],
                    'Average error in % (MAPE)': [pct(comb_test['MAPE_usd'], 2), pct(naive_test['MAPE_usd'], 2)],
                    'Direction right': [f"{pct(comb_test['DirAcc_pct'])} (p = {comb_test['DirAcc_pvalue']:.2f}; a coin flip gets 50%)", '— (makes no direction call)'],
                    'Says UP on': [f"{pct(comb_test.get('UpCalls_pct'), 0)} of days ({pct(comb_test.get('UpDays_pct'), 0)} went up)", '—'],
                    'Better than the simple guess?': [f"{'Yes' if beats else 'No'} (statistical test p = {comb_test['DM_pvalue']:.2f})", '—']})
                st.dataframe(rel_tbl, width='stretch', hide_index=True)
                st.caption("Errors: lower is better. MAE = average size of the error; RMSE = like MAE but big misses count more; "
                           "MAPE = the average error as a % of the price. More results: How Accurate Is It? tab.")
        else:
            st.info("No test results found yet. Run `make evaluate` first.")
        if fc is not None:
            with st.expander("What does each model say — and how did it do lately? (calculated when you ask)"):
                r = fc
                target = r['target_date']
                n_score = st.select_slider("Check every model on the last … days", options=[20, 30, 60, 90], value=30,
                                           on_change=lambda: st.session_state.update(open_tab=TAB_LABELS[0]),
                                           help="We test each model again on these past days and compare its guesses with the real prices.")
                score_key = (asset, r['as_of_date'], n_score)
                if st.button("Check the models now", on_click=lambda: st.session_state.update(open_tab=TAB_LABELS[0])):
                    with st.status(f"Checking {len(r['individual'])} models on the last {n_score} days…", expanded=True) as score_box:
                        st.write(f"For each of the last {n_score} days, the models only see older data, make a guess, and we check it:")

                        def _scored(model_name, secs):
                            took = f"{secs * 1000:.0f} ms" if secs < 1 else f"{secs:.1f} s"
                            st.write(f"✔ **{model_name}** — {n_score} forecasts made and checked in {took}")

                        sc_ = recent_track_record(asset, n_score, progress=_scored)
                        score_box.update(label=f"Checked {sc_['n_days']} days × {len(r['individual'])} models in {sc_['compute_seconds']:.1f} s",
                                         state="complete", expanded=False)
                    st.session_state['run_score'] = (score_key, sc_)
                if st.session_state.get('run_score', (None,))[0] != score_key:
                    st.caption("Press **Check the models now** to run every model again on recent days.")
                else:
                    sc = st.session_state['run_score'][1]
                    by_model = {row['model']: row for row in sc['rows']}
                    rw = by_model['Random walk']
                    tomorrow = {m['model']: m for m in r['individual']}
                    tomorrow[COMBINED] = {'predicted_price': r['predicted_price'], 'predicted_return_pct': r['predicted_return_pct'], 'direction': r['direction']}
                    table = []
                    for m in list(tomorrow):
                        row, pred_m = by_model[m], tomorrow[m]
                        table.append({'Model': 'Our forecast (average of all models)' if m == COMBINED else m,
                                      f"Forecast for {target}": f"{usd(pred_m['predicted_price'])} ({pred_m['predicted_return_pct']:+.2f}%)",
                                      'Direction': 'UP ▲' if pred_m['direction'] == 'UP' else 'DOWN ▼',
                                      f'Direction right (last {sc["n_days"]} days)': f"{row['hits']} of {row['n_moved']} = {row['hit_pct']:.0f}%",
                                      'Average price error': f"{row['mae_pct']:.2f}%",
                                      'Compared with "tomorrow = today"': f"{rw['mae_pct']:.2f}% → " + ('same' if abs(row['mae_pct'] - rw['mae_pct']) < 0.005 else    # equal at the shown precision
                                                                                             ('better' if row['mae_pct'] < rw['mae_pct'] else 'worse'))})
                    st.dataframe(pd.DataFrame(table), width='stretch', hide_index=True)
                    st.markdown(f"<div class='live'><span class='dot'>● LIVE</span> &nbsp;Checked at {sc['computed_at'][11:16]} on <b>{sc['n_days']} days</b> "
                                f"({sc['first_day']} → {sc['last_day']}). The models never learned from these days.</div>", unsafe_allow_html=True)
                    dd = sc['days'].iloc[::-1]
                    st.dataframe(pd.DataFrame({'Forecast made on': dd['date'], 'Forecast for': dd['target_date'], 'Price that day': dd['close'].map(usd),
                                               'Forecast': dd['predicted'].map(usd) + ' (' + dd['predicted_pct'].map(lambda v_: f"{v_:+.2f}%") + ')',
                                               'Real price': dd['actual'].map(usd) + ' (' + dd['actual_pct'].map(lambda v_: f"{v_:+.2f}%") + ')',
                                               'Direction': dd['hit'].map(lambda v_: 'RIGHT ✔' if v_ is True else ('WRONG ✘' if v_ is False else 'no change'))}),
                                 width='stretch', hide_index=True)
                    n_up = sum(m['direction'] == 'UP' for m in r['individual'])
                    st.caption(f"{n_up} of {len(r['individual'])} models say UP. Scores on a few weeks change a lot; long-term results are on the "
                               "How Accurate Is It? tab.")

        # ── ⑤ price chart
        section(5, "Price chart", f"What has {asset}'s price done, and where does our forecast sit?")
        OV_BB, OV_RSI, OV_MACD, OV_VOL = ("Bollinger Bands — usual price range", "RSI — overbought / oversold", "MACD — momentum",
                                          "Volume — amount traded")
        overlay_options = [OV_BB, OV_RSI, OV_MACD] + ([OV_VOL] if asset == 'Bitcoin' else [])
        overlays = st.multiselect("Add indicator lines (optional)", overlay_options, default=[],
                                  on_change=lambda: st.session_state.update(open_tab=TAB_LABELS[0]),
                                  help="Common indicators used by traders; our models use them as inputs. Volume is offered for Bitcoin only, "
                                       "because Yahoo's volume data for gold and silver futures is not reliable (and is not a model input).")
        show_bb, show_rsi, show_macd, show_vol = (OV_BB in overlays, OV_RSI in overlays, OV_MACD in overlays, OV_VOL in overlays)
        SPANS = {"3 months": 92, "1 year": 365, "3 years": 3 * 365, "All": None}
        span = st.radio("Time span", list(SPANS), index=1, horizontal=True,
                        on_change=lambda: st.session_state.update(open_tab=TAB_LABELS[0]))
        x_start = (df['timestamp'].max() - pd.Timedelta(days=SPANS[span])) if SPANS[span] else df['timestamp'].min()
        dfc = df[df['timestamp'] >= x_start]                  # the chart shows only this window, so the price axis fits it
        n_rows, heights = 1, [0.6]
        for flag, h in ((show_vol, 0.14), (show_rsi, 0.13), (show_macd, 0.13)):
            if flag:
                n_rows += 1; heights.append(h)
        if n_rows == 1:
            heights = [1.0]
        fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=heights)
        fig.add_trace(go.Scatter(x=dfc['timestamp'], y=dfc['price'], mode='lines', name=f'{asset} closing price',
                                 line=dict(color=COLOR[asset], width=1.8)), row=1, col=1)
        t_end, v_end = pd.Timestamp(TRAIN_END), pd.Timestamp(VAL_END)
        for x0, x1, c in ((t_end, v_end, SPLIT_VAL), (v_end, df['timestamp'].max(), SPLIT_TEST)):
            if x1 > x_start:                                  # only the shaded periods inside the shown window
                fig.add_vrect(x0=max(x0, x_start), x1=x1, fillcolor=c, opacity=0.10, line_width=0, row=1, col=1)
        for x in (t_end, v_end):
            if x > x_start:
                fig.add_vline(x=x, line=dict(color='rgba(255,255,255,0.35)', width=1, dash='dot'), row=1, col=1)
        if fc is not None:
            next_date = pd.Timestamp(fc['target_date'])
            if fc.get('uncertainty_band'):
                fig.add_trace(go.Scatter(x=[next_date, next_date], y=fc['uncertainty_band'], mode='lines', name="likely range for the next day",
                                         line=dict(color=BAND, width=8)), row=1, col=1)
            fig.add_trace(go.Scatter(x=[dfc['timestamp'].iloc[-1], next_date], y=[fc['current_price'], fc['predicted_price']],
                                     mode='lines+markers', name='our forecast', line=dict(color=ACCENT, width=2, dash='dash'),
                                     marker=dict(size=8)), row=1, col=1)
        if show_bb:
            for col, nm, dash in (('BB_Upper', 'Bollinger upper band', 'dash'), ('BB_Lower', 'Bollinger lower band', 'dash'), ('BB_Mid', '20-day average price', 'dot')):
                fig.add_trace(go.Scatter(x=dfc['timestamp'], y=dfc[col], mode='lines', name=nm, line=dict(color='gray', width=1, dash=dash)), row=1, col=1)
        row = 1
        if show_vol and 'volume' in dfc.columns:
            row += 1
            colors = np.where(dfc['price'].diff().fillna(0) >= 0, UP_COL, DOWN_COL)
            fig.add_trace(go.Bar(x=dfc['timestamp'], y=dfc['volume'], name='Volume (amount traded)', marker_color=colors, opacity=0.6), row=row, col=1)
            fig.update_yaxes(title_text="volume", row=row, col=1)
        if show_rsi:
            row += 1
            fig.add_trace(go.Scatter(x=dfc['timestamp'], y=dfc['RSI'], mode='lines', name='RSI (14 days; above 70 = overbought, below 30 = oversold)',
                                     line=dict(color='#E91E63', width=1.2)), row=row, col=1)
            fig.add_hline(y=70, line=dict(color=NAIVE, width=1, dash='dot'), row=row, col=1)
            fig.add_hline(y=30, line=dict(color=NAIVE, width=1, dash='dot'), row=row, col=1)
            fig.update_yaxes(title_text="RSI", row=row, col=1)
        if show_macd:
            row += 1
            fig.add_trace(go.Scatter(x=dfc['timestamp'], y=dfc['MACD'], mode='lines', name='MACD', line=dict(color='#2196F3', width=1.2)), row=row, col=1)
            fig.add_trace(go.Scatter(x=dfc['timestamp'], y=dfc['MACD_Signal'], mode='lines', name='MACD signal line', line=dict(color='#FF9800', width=1.2)), row=row, col=1)
            h_ = dfc['MACD'] - dfc['MACD_Signal']
            fig.add_trace(go.Bar(x=dfc['timestamp'], y=h_, name='MACD minus signal', marker_color=np.where(h_ >= 0, UP_COL, DOWN_COL)), row=row, col=1)
            fig.update_yaxes(title_text="MACD", row=row, col=1)
        fig.update_yaxes(title_text="price (USD)", tickprefix="$", row=1, col=1)
        st.markdown(f"**{asset} closing price — {span.lower() if span != 'All' else 'all data since 2018'}**", unsafe_allow_html=True)
        st.plotly_chart(layout(fig, 460 if n_rows == 1 else 460 + (n_rows - 1) * 110), use_container_width=True)
        st.caption(f"**What this shows:** {asset}'s daily closing price, with our forecast and its likely range for the next day at the right "
                   f"end. Shaded areas: purple = validation data used to tune the models ({TRAIN_END} → {VAL_END}); blue = test data the models "
                   "never learned from (after that). **What to take from it:** the likely range is wide compared with the forecast step — "
                   "one day ahead, the price can easily land anywhere inside it.")

        # ── where next
        st.markdown("<div class='next'><b>Explore further</b><br>"
                    "· <b>Test a Past Day</b> — pick a day the models never saw and check their guess against the real price.<br>"
                    "· <b>How Accurate Is It?</b> — all results against simple benchmarks, for all three assets.<br>"
                    "· <b>How It Works</b> — the data, the models, and the limitations, explained.</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════ 2. PREDICT A DAY (unseen test period)
with tab_demo:
    st.subheader("Test the models on a past day they have never seen")
    st.markdown(f"Pick any day from **{VAL_END}** onwards. These days come after the models' training data, and no model was trained or tuned "
                f"on them. The models see data **only up to the day you pick**, forecast the next trading day, and then the real price is revealed. "
                f"New days are added automatically as Yahoo Finance publishes new prices (marked **NEW**).")
    if hist is None or df is None:
        st.warning("The test results are missing. Run `make evaluate` first.")
    else:
        stored_days = list(hist['date'].dt.date)
        new_days = ([d for d in df['timestamp'].dt.date if d > stored_days[-1]] if df.attrs.get('source') == 'live download' else [])
        test_days = stored_days + new_days
        newest = test_days[-1]
        new_set = set(new_days)

        def _day_label(d):
            lab = d.strftime('%Y-%m-%d (%a)')
            if d in new_set:
                lab += '  · NEW' + ('  · next price not published yet' if d == newest else '')
            return lab

        c1, c2 = st.columns([3, 1])
        with c1:
            pick = st.selectbox("Pick a day (the models see data up to and including this day)", test_days[::-1], index=1 if new_days else 0,
                                on_change=lambda: st.session_state.update(open_tab="Test a Past Day"),
                                format_func=_day_label)
        st.caption(f"{len(stored_days)} days from the stored test period ({stored_days[0]} → {stored_days[-1]})"
                   + (f" + **{len(new_days)} newer days** from the latest download ({new_days[0]} → {new_days[-1]})" if new_days else "")
                   + f". Newest first; the latest closing price is from {df['timestamp'].iloc[-1].date()}.")
        with c2:
            st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
            go_demo = st.button("▶ Run the models now", type="primary", width='stretch',
                                help="Runs all the models now, using data up to the chosen day only, then shows the real price.",
                                on_click=lambda: st.session_state.update(open_tab="Test a Past Day"))
        stored = st.session_state.get('demo_result')
        if not go_demo and not (stored and stored[0] == asset and stored[1] == pick):
            st.info(f"Press **▶ Run the models now**: the {len(members)} models will run one after another on the data up to {pick}, "
                    "and you will see each result as it appears.")
        else:
            try:
                if go_demo:
                    with st.status(f"Running {len(members)} models for {pick}…", expanded=True) as run_box:
                        st.write(f"**① Hiding the future** — the models only get {asset} data up to **{pick}**. Every later day is removed.")
                        st.write(f"**② Running the models** on the last 30 days of data:")

                        def _show(model_name, r_hat, secs):
                            took = f"{secs * 1000:.0f} ms" if secs < 1 else f"{secs:.1f} s"
                            st.write(f"✔ **{model_name}** forecasts {r_hat * 100:+.3f}% for the next day  ·  took {took}")

                        r = predict_for_date(asset, str(pick), COMBINED, progress=_show)
                        st.write(f"**③ Averaging** the {len(r['individual'])} forecasts → **{r['predicted_return_pct']:+.3f}%** → "
                                 f"forecast closing price **{usd_md(r['predicted_price'])}**")
                        st.write(f"**④ Revealing** the real closing price of {r['actual_date'] or 'the next day'} and checking the forecast.")
                        run_box.update(label=f"Finished in {r['compute_seconds']:.1f} s ({r['computed_at'][11:]})",
                                       state="complete", expanded=True)
                    st.session_state['demo_result'] = (asset, pick, r)
                else:
                    r = stored[2]
                st.markdown(
                    f"<div class='live'><span class='dot'>● LIVE</span> &nbsp;Calculated at <b>{r['computed_at'][11:]}</b> on {r['computed_at'][:10]} in "
                    f"<b>{r['compute_seconds']:.1f} s</b> · <b>{len(r['individual'])} models</b> run · data used: "
                    f"<b>{r['history_start']} → {r['as_of_date']}</b> ({r['rows_used']:,} days; the models look at the last 30 days, starting "
                    f"{r['window_start']}; {src_txt(r['data_source'])}) · <b>{r['rows_hidden']:,} later day{'s' if r['rows_hidden'] != 1 else ''} hidden</b> "
                    f"from the models. Press ▶ again to recalculate.</div>",
                    unsafe_allow_html=True)
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric(f"Closing price on {r['as_of_date']}", usd(r['current_price']))
                m2.metric(f"Forecast for {r['actual_date'] or 'the next day'}", usd(r['predicted_price']), f"{r['predicted_return_pct']:+.2f}%")
                if r['actual_price'] is not None:
                    m3.metric("Real closing price", usd(r['actual_price']), f"{r['actual_return_pct']:+.2f}%")
                    m4.metric("Forecast error", f"{r['error_pct']:+.2f}%", f"{r['error_usd']:+,.2f} USD", delta_color="off")
                    hit = r['direction_hit']
                    m5.metric("Direction", f"{r['direction']} → {'RIGHT ✔' if hit else ('WRONG ✘' if hit is not None else 'no change')}")
                    b = r['uncertainty_band']
                    st.caption(f"Likely range (68%) for that day: **{usd_md(b[0])} – {usd_md(b[1])}** (±{r['band_halfwidth_pct']:.2f}% = {r['band_k']:.2f} × "
                               f"the expected daily move known on {r['as_of_date']}). The real price landed **{'inside' if r['in_band'] else 'outside'}** it.")
                else:
                    m3.metric("Real closing price", "not published yet")
                period_txt = ('after the stored test period (newer data, never checked before)' if r.get('after_stored_test') else
                              'inside the stored test period (days the models never saw)' if r['in_unseen_test_period'] else 'inside the training / validation period')
                st.caption(f"Forecast: **average** of {len(r['individual'])} models · for the next {r['horizon_days']} trading day · {period_txt} · "
                           f"uses only data up to the chosen day ({src_txt(r['data_source'])}).")

                col_pred = f'pred_close_{COMBINED}'
                if col_pred in hist.columns:
                    fd = go.Figure()
                    fd.add_trace(go.Scatter(x=hist['target_date'], y=hist['actual_close'], mode='lines', name='real closing price',
                                            line=dict(color=COLOR[asset], width=2)))
                    after = df[df['timestamp'] > hist['target_date'].max()] if r.get('after_stored_test') else df.iloc[:0]
                    if len(after):                                       # newer live closes, so a NEW day's markers are not left floating
                        fd.add_trace(go.Scatter(x=pd.concat([hist['target_date'].tail(1), after['timestamp']]),
                                                y=pd.concat([hist['actual_close'].tail(1), after['price']]), mode='lines',
                                                name='real closing price (newer data)', line=dict(color=COLOR[asset], width=2, dash='dot')))
                    fd.add_trace(go.Scatter(x=hist['target_date'], y=hist[col_pred], mode='lines', name='forecast closing price',
                                            line=dict(color=ACCENT, width=1.4, dash='dash')))
                    if f'band_lo_{COMBINED}' in hist.columns:
                        fd.add_trace(go.Scatter(x=pd.concat([hist['target_date'], hist['target_date'][::-1]]),
                                                y=pd.concat([hist[f'band_hi_{COMBINED}'], hist[f'band_lo_{COMBINED}'][::-1]]),
                                                fill='toself', fillcolor='rgba(0,224,198,0.12)', line=dict(width=0), name="likely range (68%)", hoverinfo='skip'))
                    if r['actual_price'] is not None:
                        fd.add_trace(go.Scatter(x=[pd.Timestamp(r['actual_date'])], y=[r['predicted_price']], mode='markers', name='this forecast',
                                                marker=dict(color=ACCENT, size=13, symbol='diamond', line=dict(color='white', width=1))))
                        fd.add_trace(go.Scatter(x=[pd.Timestamp(r['actual_date'])], y=[r['actual_price']], mode='markers', name='real price',
                                                marker=dict(color=COLOR[asset], size=13, symbol='circle', line=dict(color='white', width=1))))
                    fd.update_yaxes(tickprefix="$")
                    st.plotly_chart(layout(fd, 400, f"Test period {hist['target_date'].min().date()} → {hist['target_date'].max().date()}: "
                                                    f"real vs forecast closing price" + (" (+ newer prices)" if len(after) else "")),
                                    use_container_width=True)
                    st.caption("The forecast line always stays close to the real line, because each forecast starts from the previous day's price. "
                               "So this chart cannot show whether the models have skill — that needs the % changes (How Accurate Is It? tab).")

                if r['individual']:
                    t = pd.DataFrame(r['individual'])
                    t['Forecast'] = t['predicted_price'].map(usd)
                    t['Change'] = t['predicted_return_pct'].map(lambda v: f"{v:+.2f}%")
                    t['Error'] = t['error_pct'].map(lambda v: f"{v:+.2f}%" if pd.notna(v) else "—")
                    t['Direction'] = t['direction_hit'].map(lambda v: 'RIGHT ✔' if v is True else ('WRONG ✘' if v is False else '—'))
                    with st.expander("Each model on this day (their average is the forecast above)"):
                        st.dataframe(t[['model', 'Forecast', 'Change', 'Error', 'Direction']].rename(columns={'model': 'Model'}),
                                     width='stretch', hide_index=True)
            except Exception as e:
                st.error(f"Sorry, the forecast could not be made: {e}")
                logger.exception("Demo prediction error")

# ═══════════════════════════════════════════════ 3. MODEL PERFORMANCE
with tab_perf:
    if cv_all is None or te_all is None or not tm:
        st.warning("The results are missing. Run `make evaluate` to create them.")
    else:
        s = status
        t = te_all[te_all['asset'] == asset].sort_values('RMSE_ret')
        is_naive = (t['model'] == 'Naive').values
        BASE_LBL = {'Naive': 'Simple guess: tomorrow = today (random walk)', 'Naive-Mean': 'Long-term average move (drift)',
                    'ARIMA': 'ARIMA (classic statistics model)', 'MA-5': 'Average of the last 5 prices (MA-5)',
                    'Mean-20d': 'Average move of the last 20 days (Mean-20d)'}
        label = lambda m_: ("Our forecast (average of 6 models) ★" if m_ == COMBINED else
                            (f"{m_} (best single model in back-tests)" if m_ == cv_best else BASE_LBL.get(m_, m_)))
        show = pd.DataFrame({
            'Model': t['model'].map(label),
            'Average error $ (MAE)': t['MAE_usd'].map(usd), 'Typical error $ (RMSE)': t['RMSE_usd'].map(usd),
            'Average error % (MAPE)': t['MAPE_usd'].map(lambda v: pct(v, 2)),
            'Typical error of the % move (RMSE)': t['RMSE_ret'].map(lambda v: f"{v:.5f}"), 'Move explained (R²)': t['R2_ret'].map(lambda v: f"{v:+.3f}"),
            'Direction right (p)': np.where(is_naive, '—', t['DirAcc_pct'].map(lambda v: pct(v)) + ' (p ' + t['DirAcc_pvalue'].map(lambda v: f"{v:.2f}") + ')'),
            'Says UP on': np.where(is_naive, '—', t['UpCalls_pct'].map(lambda v: pct(v, 0))) if 'UpCalls_pct' in t.columns else '—',
            'Better than simple guess? (p)': np.where(is_naive, '—', t['DM_pvalue'].map(lambda v: f"{v:.2f}")),
            'Buy only when forecast is UP': t['strategy_return_pct'].map(lambda v: f"{v:+.1f}%"),
            'Buy and hold': t['buy_hold_return_pct'].map(lambda v: f"{v:+.1f}%")})
        st.markdown("<div class='intro'>This tab shows how well the system did on days it never learned from, compared with simple rules. "
                    "The short answer comes first; the full details for examiners are in the sections at the bottom.</div>", unsafe_allow_html=True)
        st.subheader("1 · The short answer")
        naive_r, best = t[is_naive].iloc[0], t[t['model'] == COMBINED].iloc[0]
        rel = (best['RMSE_ret'] / naive_r['RMSE_ret'] - 1) * 100
        if best['DM_pvalue'] < 0.05 and rel < 0:
            verdict = f"it beats the simple guess ({rel:+.2f}% error; statistical test p = {best['DM_pvalue']:.2f}, so the difference is real)"
        elif abs(rel) < 0.05:
            verdict = f"it is about the same as the simple guess (statistical test p = {best['DM_pvalue']:.2f})"
        else:
            verdict = (f"its error is {abs(rel):.2f}% {'smaller' if rel < 0 else 'larger'} than the simple guess — too small a difference to be "
                       f"real (statistical test p = {best['DM_pvalue']:.2f}; it would need to be below 0.05)")
        da_sig = 'better than chance' if best['DirAcc_pvalue'] < 0.05 else 'not better than chance'
        up_txt = (f" The forecast said UP on {best['UpCalls_pct']:.0f}% of the days ({best['UpDays_pct']:.0f}% of days actually went up), so its direction "
                  f"mostly follows {asset}'s long-term rise, not day-to-day timing." if 'UpCalls_pct' in best and pd.notna(best['UpCalls_pct']) else '')
        st.markdown(f"<div class='verdict'><b>Honest reading.</b> On the test days, our forecast's typical error (RMSE of the daily % move) is {best['RMSE_ret']:.5f}, against "
                f"{naive_r['RMSE_ret']:.5f} for the simple guess \"tomorrow = today\" — {verdict}. It got the direction right on "
                f"{best['DirAcc_pct']:.1f}% of {int(best['DirAcc_n'])} days (p = {best['DirAcc_pvalue']:.2f}, {da_sig}).{up_txt} "
                "Note: a low average % error (MAPE) does not mean the forecast is good — the simple guess gets the same score. "
                "R² on prices is not shown on purpose: even the simple guess scores about 0.95 there.</div>", unsafe_allow_html=True)

        # comparison chart: every model vs the random walk
        tt = t[~t['model'].isin(ROLLING_BASELINES)].copy()      # the moving-average baselines are far off the zoomed scale (see caption)
        tt['label'] = tt['model'].map(lambda m_: {COMBINED: 'Our forecast ★', 'Naive': 'Simple guess', 'Naive-Mean': 'Drift'}.get(m_, m_))
        fcmp = make_subplots(rows=1, cols=2, subplot_titles=("Typical error of the daily % move (lower is better)", "Direction right, % (higher is better)"),
                           horizontal_spacing=0.08)
        colors = [NAIVE if m_ == 'Naive' else (ACCENT if m_ == COMBINED else '#4E7FA8') for m_ in tt['model']]
        fcmp.add_trace(go.Bar(x=tt['label'], y=tt['RMSE_ret'], marker_color=colors, name='typical error', showlegend=False), row=1, col=1)
        fcmp.add_hline(y=float(naive_r['RMSE_ret']), line=dict(color=NAIVE, dash='dash', width=1.5), row=1, col=1,
                     annotation_text="simple guess (tomorrow = today)", annotation_position="top left", annotation_font_color=NAIVE)
        nn = tt[tt['model'] != 'Naive']
        fcmp.add_trace(go.Bar(x=nn['label'], y=nn['DirAcc_pct'], marker_color=[ACCENT if m_ == COMBINED else '#4E7FA8' for m_ in nn['model']],
                            name='direction right %', showlegend=False), row=1, col=2)
        fcmp.add_hline(y=50, line=dict(color=NAIVE, dash='dash', width=1.5), row=1, col=2, annotation_text="coin flip (50%)",
                     annotation_position="top left", annotation_font_color=NAIVE)
        fcmp.update_yaxes(range=[float(tt['RMSE_ret'].min()) * 0.985, float(tt['RMSE_ret'].max()) * 1.01], row=1, col=1)
        fcmp.update_yaxes(range=[40, 65], row=1, col=2)
        fcmp.update_layout(hovermode="closest")
        st.plotly_chart(layout(fcmp, 340, "Every model vs the simple guess — test days", legend=False), use_container_width=True)
        spread = float((tt['RMSE_ret'] / float(naive_r['RMSE_ret']) - 1).abs().max() * 100)
        ma = t[t['model'].isin(ROLLING_BASELINES)]
        ma_txt = ('' if ma.empty else ' Not shown: the moving-average rules (' + ', '.join(
            f"{m_} {(v / float(naive_r['RMSE_ret']) - 1) * 100:+.0f}%" for m_, v in zip(ma['model'], ma['RMSE_ret'])) +
            " error compared with the simple guess). A moving average reacts late to price changes, so it is a much weaker benchmark than "
            "\"tomorrow = today\".")
        st.caption(f"**What this shows:** left, each model's typical error (lower is better); right, how often each got the direction "
                   f"right (higher is better). **What to take from it:** all the bars are almost the same height as the simple guess and the "
                   f"coin-flip line. ★ = our forecast. The error axis is zoomed in: every model, the simple guess, the drift and ARIMA are all within "
                   f"±{spread:.1f}% of the simple guess.{ma_txt}")

        if te_all is not None:
            st.subheader("2 · All three assets")
            st.caption("The same check for Bitcoin, Gold and Silver: is our forecast better than simple rules?")
            rows_x = []
            for a_ in ASSETS:
                ta = te_all[te_all['asset'] == a_].set_index('model')
                if COMBINED not in ta.index or 'Naive' not in ta.index:
                    continue
                rw_ = float(ta.loc['Naive', 'RMSE_ret'])
                rel_ = lambda m_: f"{(float(ta.loc[m_, 'RMSE_ret']) / rw_ - 1) * 100:+.2f}%" if m_ in ta.index else '—'
                cvb = status_all.get(a_, {}).get('primary_model')
                rows_x.append({'Asset': a_, 'Test days': int(ta.loc[COMBINED, 'n_test']), 'Simple guess: typical error': f"{rw_:.5f}",
                               'Our forecast': rel_(COMBINED), 'Best single model': f"{cvb}: {rel_(cvb)}" if cvb else '—',
                               'Drift': rel_('Naive-Mean'), 'ARIMA': rel_('ARIMA'),
                               **{b: rel_(b) for b in ROLLING_BASELINES},
                               'Our forecast: test p': f"{float(ta.loc[COMBINED, 'DM_pvalue']):.2f}",
                               'Our forecast: direction right (says UP on)': f"{float(ta.loc[COMBINED, 'DirAcc_pct']):.1f}% ({float(ta.loc[COMBINED, 'UpCalls_pct']):.0f}%)"})
            if rows_x:
                st.dataframe(pd.DataFrame(rows_x), width='stretch', hide_index=True)
                st.caption("Each cell shows how much larger (+) or smaller (−) that forecast's typical error is than the simple guess's, on exactly "
                           "the same days. None of the learned forecasts is clearly better than the simple guess (all p ≥ 0.05), while the "
                           "moving-average rules are much worse. That is why \"tomorrow = today\" is the benchmark that matters.")

        vt, vw = load_csv(os.path.join('volatility', 'vol_test.csv')), load_csv(os.path.join('volatility', 'vol_walkforward.csv'))
        if vt is not None and vw is not None and len(vt[vt['asset'] == asset]):
            st.subheader("3 · Risk forecast — the part that works")
            st.caption("Can we guess how BIG the next moves will be? For Bitcoin and Gold: yes, clearly better than a simple rule.")
            vta = vt[vt['asset'] == asset].sort_values('RMSE_log')
            chosen = vta[vta['selected']]['candidate'].iloc[0] if vta['selected'].any() else '—'
            st.dataframe(pd.DataFrame({
                'Forecast method': vta['candidate'].map(lambda c: f"{c} (used in the app)" if c == chosen else c),
                'Error (RMSE of log volatility)': vta['RMSE_log'].map(lambda v: f"{v:.4f}"),
                'vs "same swings as the last 5 days"': vta['RMSE_vs_ref_pct_vs_persistence'].map(lambda v: f"{v:+.1f}%"),
                'p': vta['DM_p_vs_persistence'].map(lambda v: '—' if pd.isna(v) else f"{v:.3f}"),
                'vs simple average (EWMA)': vta['RMSE_vs_ref_pct_vs_ewma'].map(lambda v: f"{v:+.1f}%"),
                'p (vs EWMA)': vta['DM_p_vs_ewma'].map(lambda v: '—' if pd.isna(v) else f"{v:.3f}")}), width='stretch', hide_index=True)
            # walk-forward evidence of the SERVED candidate (HAR-IV is scored on the days where its implied-volatility series exists)
            vwa = vw[(vw['asset'] == asset) & (vw['candidate'] == chosen)]
            best_wf = (vwa[vwa['days'] != 'all out-of-fold'] if chosen == 'HAR-IV' else vwa[vwa['days'] == 'all out-of-fold']).iloc[0]
            st.caption(f"Test period ({int(vta['n'].iloc[0])} days). Here we forecast the size of price swings (volatility) over the next 5 trading "
                       f"days; negative % = smaller error. The method used in the app (**{chosen}**) was chosen BEFORE the test days were looked at, "
                       f"using a rule written down in advance (`results/volatility/PREREGISTRATION.md`), on {int(best_wf['n']):,} back-test days where "
                       f"it was {abs(best_wf['RMSE_vs_ref_pct']):.0f}% better than \"same swings as the last 5 days\". The other rows are shown only "
                       "for transparency — choosing the best test row afterwards would be cheating."
                       + (f" In this test period the chosen {chosen} method did **not** beat \"same swings as the last 5 days\" "
                          f"({float(vta[vta['selected']]['RMSE_vs_ref_pct_vs_persistence'].iloc[0]):+.1f}%). We report this instead of hiding it, and we "
                          "do not switch to a better-looking row, because it was not the method chosen in advance."
                          if vta['selected'].any() and float(vta[vta['selected']]['RMSE_vs_ref_pct_vs_persistence'].iloc[0]) >= 0 else ''))

        bc = load_csv('band_calibration.csv')
        if bc is not None and len(bc[bc['asset'] == asset]):
            st.subheader("4 · Likely price range — is it honest?")
            st.caption("Does the real price land inside our 68% range about 2 out of 3 times, as promised?")
            b = bc[bc['asset'] == asset]
            k_ = float(b['k'].iloc[0]) if 'k' in b.columns else float('nan')
            sig_m = (status.get('band') or {}).get('sigma_method', 'EWMA')
            names = {'conditional': f'used in the app: ±{k_:.2f} × σ (σ = expected daily move, {sig_m} method)',
                     'conditional_1sigma': f'±1 × σ ({sig_m} method, not adjusted)',
                     'fixed': 'fixed width (ignores market conditions)',
                     'conditional_ewma': 'previous range: k × σ with the EWMA method'}
            split_txt = {'validation': 'validation (tuning period)', 'test': 'test (days never seen)'}
            st.dataframe(pd.DataFrame({'Period': b['split'].map(lambda v: split_txt.get(v, v)), 'Range type': b['band'].map(lambda v: names.get(v, v)),
                                       'Days': b['n'], 'Real price inside the range (target 68%)': b['coverage_pct'].map(lambda v: f"{v:.1f}%"),
                                       'Average half-width': b['mean_sigma_pct'].map(lambda v: f"{v:.2f}%")}),
                         width='stretch', hide_index=True)
            one_sig = b.loc[(b['band'] == 'conditional_1sigma') & (b['split'] == 'test'), 'coverage_pct']
            one_sig_txt = (f" a plain ±1σ range contained the price on {float(one_sig.iloc[0]):.1f}% of test days — too often, so it was too wide."
                           if len(one_sig) else "")
            st.caption(f"A 68% range should contain the real price on about 68% of days (about 2 out of 3). The multiplier k = {k_:.2f} and the fixed "
                       "width were set using errors from the training years only, so the validation and test rows are a fair check. Very big surprise "
                       "days happen more often than a normal bell curve suggests (\"fat tails\"):"
                       f"{one_sig_txt} The range in the app uses only the information available on the day of the forecast, so it widens when "
                       "markets are turbulent and narrows when they are calm. This is where the data shows real skill: the size of tomorrow's move "
                       "is much more predictable than its direction.")

        st.subheader("5 · Full details (for examiners)")
        st.caption("Every table behind the results above. Open only what you need.")
        with st.expander("Full test results — every model and every simple benchmark"):
            st.caption(f"Test period {s['test_period'][0]} → {s['test_period'][1]} ({tm['n_test']} days). The forecast shown in the app is the "
                       f"**Our forecast** row: a plain average of the six models, with no weights fitted, so nothing was tuned on these test days. "
                       f"The best single model in back-tests is **{cv_best}**. Honest note: we did look at this period during development (for example "
                       f"when we found and fixed a data-timing error, and when we added the average and the likely range), so it is not a perfectly "
                       f"untouched test. The ~6-year back-test record below, and the newer days after the data was frozen, are cleaner evidence.")
            st.dataframe(show, width='stretch', hide_index=True)
            st.caption("How to read this table: **errors** — lower is better. **Move explained (R²)** — how much of the daily move the forecast "
                       "explains (0 = none, 1 = all). **p** — a p-value below 0.05 means the result is unlikely to be luck. **Better than simple guess? (p)** "
                       "— a statistical test (Diebold–Mariano) comparing the errors with \"tomorrow = today\". **Buy only when forecast is UP** — what "
                       "you would have made by holding the asset only on days the forecast said UP (0.1% cost per trade), next to simply buying and holding.")

        with st.expander("Back-tests on past years (walk-forward) — how the models were chosen"):
            c = cv_all[cv_all['asset'] == asset].sort_values('RMSE_ret_mean')
            showc = pd.DataFrame({'Model': c['model'].map(label),
                                  'Typical error of the % move (average ± spread over 4 back-tests)': c['RMSE_ret_mean'].map(lambda v: f"{v:.5f}") + ' ± ' + c['RMSE_ret_std'].map(lambda v: f"{v:.5f}"),
                                  'Average error of the % move': c['MAE_ret_mean'].map(lambda v: f"{v:.5f}"), 'Move explained (R²)': c['R2_ret_mean'].map(lambda v: f"{v:+.3f}"),
                                  'Direction right': np.where(c['model'] == 'Naive', '—', c['DirAcc_pct_mean'].map(lambda v: pct(v))),
                                  'Typical error ($)': c['RMSE_usd_mean'].map(usd)})
            st.dataframe(showc, width='stretch', hide_index=True)
            cl = c[~c['model'].isin(ROLLING_BASELINES)]
            cv_spread = float((cl['RMSE_ret_mean'] / float(c.loc[c['model'] == 'Naive', 'RMSE_ret_mean'].iloc[0]) - 1).abs().max() * 100)
            st.caption("Four back-tests inside the training and validation years: each time, the models learn from the past and are tested on the next "
                       "block of time (the test period is never used here). The \"Our forecast\" row averages the models' forecasts. Every model is within "
                       f"±{cv_spread:.2f}% of the simple guess"
                       f"{' (except the moving-average rules, which are much worse)' if len(cl) < len(c) else ''}, so the ranking is not reliable. "
                       "That is why the app shows the plain average of all models instead of picking one 'winner'.")

        with st.expander("Long-term evidence — about 6 years of back-test forecasts"):
            pw_all = load_csv('walkforward_pooled.csv')
            if pw_all is not None and len(pw_all[pw_all['asset'] == asset]):
                pw = pw_all[pw_all['asset'] == asset].sort_values('RMSE_ret')
                p0 = pw.iloc[0]
                st.markdown(f"**{int(p0['n_days']):,} back-test forecasts ({p0['first_day']} → {p0['last_day']})**")
                st.dataframe(pd.DataFrame({
                    'Model': pw['model'].map(label),
                    'Error vs simple guess': pw['RMSE_vs_naive_pct'].map(lambda v: f"{v:+.3f}%"),
                    'Test p vs simple guess': pw['DM_pvalue'].map(lambda v: f"{v:.3f}"),
                    'Test p vs drift': pw['DM_pvalue_vs_drift'].map(lambda v: '—' if pd.isna(v) else f"{v:.3f}"),
                    'Direction right (p)': pw['DirAcc_pct'].map(lambda v: pct(v)) + ' (p ' + pw['DirAcc_pvalue'].map(lambda v: f"{v:.3f}") + ')',
                    'Says UP on': pw['UpCalls_pct'].map(lambda v: pct(v, 0))}), width='stretch', hide_index=True)
                st.caption(f"All back-test forecasts pooled into one record ({p0['UpDays_pct']:.0f}% of days went up) — about ten times more days than the "
                           "test period, so even small real effects would show here. 'Drift' always predicts the long-term average move, which means "
                           "'always UP' for an asset that rose. To show real timing skill, a model must beat drift, not just 50%. The model settings were "
                           "tuned on these same back-tests, so if anything this favours the models. The tables hold about 30 direction tests across the "
                           "three assets, so one or two p-values below 0.05 are expected by luck alone (a strict cut-off for this many tests, the "
                           "Bonferroni threshold, is about 0.002).")

        with st.expander("Forecast vs real daily moves — charts"):
            # predicted vs actual in return space for the combined forecast
            if hist is not None and f'pred_return_{COMBINED}' in hist.columns:
                pr, ar = hist[f'pred_return_{COMBINED}'] * 100, hist['actual_return'] * 100
                f1, f2 = st.columns([3, 2])
                with f1:
                    fr = go.Figure()
                    fr.add_trace(go.Scatter(x=hist['target_date'], y=ar, name='real move', line=dict(color='rgba(255,255,255,0.75)', width=1)))
                    fr.add_trace(go.Scatter(x=hist['target_date'], y=pr, name='our forecast', line=dict(color=ACCENT, width=1.8)))
                    fr.update_yaxes(ticksuffix="%")
                    st.plotly_chart(layout(fr, 340, "Daily % moves: real vs forecast"), use_container_width=True)
                with f2:
                    sc = go.Figure()
                    sc.add_trace(go.Scatter(x=ar, y=pr, mode='markers', marker=dict(color=ACCENT, size=6, opacity=0.7), name='days'))
                    lim = float(np.abs(ar).max()) * 1.05
                    sc.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode='lines', line=dict(color='red', dash='dash', width=1), name='perfect forecast'))
                    sc.update_xaxes(title_text="real move (%)", range=[-lim, lim]); sc.update_yaxes(title_text="forecast move (%)", range=[-lim, lim])
                    sc.update_layout(hovermode="closest")
                    st.plotly_chart(layout(sc, 340, "Each dot is one day (forecast vs real)"), use_container_width=True)
                if comb_test is not None:
                    st.caption(f"The forecast line hardly moves (typical size {comb_test['pred_std'] * 100:.2f}% against {comb_test['true_std'] * 100:.2f}% for "
                               "the real moves): the models have learned that tomorrow's move is mostly unpredictable, so they stay close to \"no change\".")

        with st.expander("Which inputs do the models rely on most?"):
            # feature importance (tree / linear members)
            fi = load_csv('feature_importance.csv')
            if fi is not None:
                fi_models = [m_ for m_ in ('CatBoost', 'LightGBM', 'RandomForest', 'Ridge') if len(fi[(fi['asset'] == asset) & (fi['model'] == m_)])]
                if fi_models:
                    pick_m = st.selectbox("Which inputs does a model rely on most?", fi_models, index=0,
                                          on_change=lambda: st.session_state.update(open_tab="How Accurate Is It?"),
                                          help="GRU and LSTM (neural networks) cannot show this. Our forecast averages all six models.")
                    f_sel = fi[(fi['asset'] == asset) & (fi['model'] == pick_m)].sort_values('importance', ascending=False)
                    top = f_sel.head(15).iloc[::-1]
                    fb = go.Figure(go.Bar(x=top['importance'], y=top['feature'], orientation='h', marker_color=ACCENT))
                    fb.update_layout(hovermode="closest")
                    st.plotly_chart(layout(fb, 420, f"{pick_m}: the 15 most important of its {len(f_sel)} inputs (importance scaled to add up to 1)", legend=False), use_container_width=True)

        reg = load_csv('regime_analysis.csv')
        if reg is not None:
            with st.expander("Results in different market conditions (our forecast vs the simple guess, test period)"):
                rg = reg[(reg['asset'] == asset) & (reg['regime_type'] != 'day_regime')]
                cond_txt = {'low volatility': 'calm market (small price swings)', 'medium volatility': 'normal market',
                            'high volatility': 'turbulent market (big price swings)', 'up-trend (20d)': 'rising (last 20 days)',
                            'down-trend (20d)': 'falling (last 20 days)'}
                st.dataframe(pd.DataFrame({'Market condition': rg['regime'].map(lambda v: cond_txt.get(v, v)), 'Days': rg['n_days'],
                                           'Average error % (our forecast)': rg['mae_pct_served'].map(lambda v: f"{v:.2f}"),
                                           'Average error % (simple guess)': rg['mae_pct_naive'].map(lambda v: f"{v:.2f}"),
                                           'Direction right (our forecast)': rg['dir_hit_served_pct'].map(lambda v: pct(v))}),
                             width='stretch', hide_index=True)
                st.caption("Market conditions are measured on the day the forecast is made (size of price swings over the last 30 days, split into "
                           "three equal groups; direction of the last 20 days), so they never use future information.")
        with st.expander("Design experiments (tested on validation data only; the test period was never used)"):
            st.caption("Technical tables for examiners: each experiment tested one design idea. Column names are the raw names from the result files.")
            for fname, title in (('E1_data_size.csv', 'E1 — Does more training data help? (3 years vs all data, same validation period)'),
                                 ('E2_feature_ablation.csv', 'E2 — Removing groups of inputs one at a time (back-tests)'),
                                 ('E3_horizon.csv', 'E3 — Forecasting 1 day vs 5 days ahead'),
                                 ('E4_volatility.csv', 'E4 — Forecasting the size of price swings over 22 days (volatility)'),
                                 ('E5_audit_candidates.csv', 'E5 — Which models to average, removing duplicate inputs, and a no-drift forecast'),
                                 ('E6_final_candidates.csv', 'E6 — A volatility-adjusted target and a robust (Huber) error measure')):
                ex = load_csv(os.path.join('experiments', fname))
                if ex is not None:
                    st.markdown(f"**{title}**")
                    st.dataframe(ex[ex['asset'] == asset].round(4), width='stretch', hide_index=True)
        with st.expander("Training checks: over-fitting gap and GRU / LSTM learning curves"):
            for name in ('overfitting_gap.png', f'{get_prefix(asset)}_loss_curves.png'):
                p = os.path.join(FIGURES_DIR, name)
                if os.path.exists(p):
                    st.image(p, width='stretch')

# ═══════════════════════════════════════════════ 4. METHODOLOGY
with tab_meth:
    st.subheader("How it works")
    band_m = (status.get('band') or {})
    band_k_txt = f"{band_m['k']:.2f} for {asset}" if band_m.get('k') else "set by the evaluation script"
    cov1 = band_m.get('test_conditional_1sigma_coverage_pct')
    cov1_txt = (f"a plain ±1σ range contained {cov1:.1f}% of {asset}'s test days instead of 68%" if cov1 is not None
                else "a plain ±1σ range contains the price too often")
    close_rule = ("Gold and Silver prices are fixed at 1:30 pm New York time (the COMEX settlement), before the US stock, interest-rate and "
                  "currency markets close. So for gold and silver we only use the *previous* day's values of those markets, and the previous day's "
                  "high/low range — never information that appears after the price is fixed."
                  if asset != 'Bitcoin' else
                  "Bitcoin's daily price closes at midnight UTC, after all US markets have closed, so the same day's market values are already "
                  "known and can be used.")
    st.markdown(f"""
**What we forecast.** Tomorrow's % change in price, written as `y_t = ln(P_(t+1) / P_t)` (the "log return"). It is turned into a price with
`P̂_(t+1) = P_t · exp(ŷ_t)`. Forecasting the change, not the price itself, keeps the numbers comparable over the years and makes the simplest
benchmark clear: "no change" (ŷ = 0), i.e. tomorrow's price = today's price. This benchmark is also called the *random walk*.

**Data.** Daily prices from Yahoo Finance since January 2018. The first training example is {status.get('data_start', '2018')} (the indicators
need some history first, and the models look at 30 days at a time): {status.get('n_train', '?')} training days and {status.get('n_val', '?')}
validation days. Only **finished** days are used — the still-running day is ignored.
**How the data is split** — by date, never shuffled: training data up to {TRAIN_END} · validation data (used to tune the models) up to
{VAL_END} · test data after that, which the models never learn from. Scaling of the inputs is learned from the training data only. Model
settings are chosen with 4 back-tests inside the training and validation years ("walk-forward validation"), and the test data is used only
once, by the final evaluation script.

**Inputs ({len(status.get('features', []))} measurements).** All built from past data only: today's and earlier daily % changes, the 20-day
trend, recent price swings (several volatility measures), common trading indicators (RSI, ADX, ROC, MACD, moving-average ratios, Bollinger
bands, ATR, the daily high–low range), the daily % changes of other markets (S&P 500, VIX{', the US dollar index, oil and the 10-year interest rate' if asset != 'Bitcoin' else ''}){', the gold price' if asset == 'Silver' else ''}{', plus the crypto Fear & Greed index, the day of the week and the change in trading volume' if asset == 'Bitcoin' else ''}.
Raw prices are shown on the charts but are never given to a model. **Timing rule:** {close_rule}

**Models.** Six models learn from the data — Ridge (a linear model), Random Forest, LightGBM and CatBoost (models made of many decision trees),
and GRU and LSTM (neural networks that read the last 30 days as a sequence). They are compared with simple rules that learn nothing: the
no-change guess, the long-term average move ("drift"), ARIMA (a classic statistics model), the average of the last 5 prices and the average
move of the last 20 days. A combined "stacked" model was also tested as an experiment. Everything is scored on exactly the same days.

**The forecast shown — an average.** *Run prediction* runs all six models on the same 30 days of data and takes the **plain average** of their
forecasts; whether the average is above or below zero gives UP or DOWN. Next to it the app shows the **signal strength**: the forecast move
divided by a normal day's move σ_t. It is called *weak* below {WEAK_SIGNAL_BELOW:.2f}× (a fixed rule of thumb, not tuned on any result),
because a move that small cannot be told apart from random daily ups and downs. A plain average is a standard, robust way to combine similar
models, and no weights are fitted, so nothing was tuned on the test days. The average is tested exactly like every single model (row
*Our forecast*). The best single model in the back-tests ({cv_best}) is shown for comparison.

**Likely price range.** Every forecast comes with a 68% range `P_t · exp(ŷ_t ± k·σ_t)` — the price should land inside it on about 2 out of 3
days. σ_t is the expected size of a normal daily move, known on the day of the forecast — the **{(status.get('band') or {}).get('sigma_method', 'EWMA')}**
method for {asset} (EWMA, a weighted average of recent price swings, unless GARCH(1,1) or HAR did better in a comparison planned in advance).
It uses only past data. The multiplier `k` ({band_k_txt}) is chosen so that the range would have contained 68% of the past errors in the
training years. Very big surprise days happen more often than a normal bell curve suggests, so {cov1_txt}. How often the real price
landed inside the range is shown on the How Accurate Is It? tab.

**Risk forecast (volatility).** A second forecast predicts the *size* of the coming moves: how much the price is likely to swing over the next
5 trading days. The candidates — "same as the last 5 days", EWMA, the HAR model (Corsi 2009), HAR plus a market fear index (CBOE GVZ for
gold and silver, taken from the previous day; Deribit DVOL for Bitcoin) and three of our models (Ridge, LightGBM, CatBoost) — were compared
in back-tests using a rule written down before the test (`results/volatility/PREREGISTRATION.md`). The winner for each asset is used in
the app and was then checked once on the test days. The direction of daily moves is hard to predict, but calm and turbulent periods tend to
last for a while, so the size of the moves *can* be forecast.

**How results are measured.** Average and typical errors in dollars and in % (what a user sees) · errors and R² of the daily % change (what is
actually forecast; R² shows how much of the move is explained) · how often the direction is right, with a p-value against a 50% coin flip,
and how often the forecast says UP (to reveal the "always UP" effect) · a statistical test of the errors against the no-change guess
(Diebold–Mariano; a p-value below 0.05 means a real difference) · a simple trading check (hold only when the forecast says UP, 0.1% cost
per trade) · how often the likely range contained the real price. The same tests are also run on all back-test days pooled together
(about 6 years), which is far more data than the test period alone.
""")
    st.markdown("**Limitations — what this system cannot do**")
    _up = {a_: float(te_all[(te_all['asset'] == a_) & (te_all['model'] == COMBINED)]['UpCalls_pct'].iloc[0])
           for a_ in ('Gold', 'Silver') if te_all is not None and len(te_all[(te_all['asset'] == a_) & (te_all['model'] == COMBINED)])}
    _n = [int(status_all.get(a_, {}).get('combined_test', {}).get('n_test', 0)) for a_ in ASSETS]
    st.markdown(
        "- **No reliable next-day direction.** For all three assets the forecast was not better than \"tomorrow's price = today's price\".\n"
        + (f"- **Gold and Silver lean UP.** The models said UP on {_up.get('Gold', float('nan')):.0f}% (Gold) and "
           f"{_up.get('Silver', float('nan')):.0f}% (Silver) of test days, because both rose in the training years — so in a falling "
           "market the direction is often wrong.\n" if _up else "")
        + (f"- **Short test period.** {min(_n)}–{max(_n)} test days, and we looked at them during development. The ~6-year back-tests "
           "and the newer days after the data was frozen give the same answer.\n" if all(_n) else "")
        + "- **The risk forecast tells size, not direction.** Single shock days can be several times bigger than expected.\n"
        "- **Free data.** Yahoo Finance prices; gold and silver follow a futures contract that is switched over time, so a few days can look odd.\n"
        "- **Not financial advice.** This is a university research project.")
    st.markdown("**The forecast shown — details**")
    fit = status.get('fit_info') or {}
    stop = (f"{fit['epochs_used']} training rounds (epochs)" if 'epochs_used' in fit else
            f"{fit.get('iterations_used', fit.get('n_estimators_used', '—'))} trees (boosting rounds)" if fit else '—')
    details = pd.DataFrame({'Item': ['Forecast shown', 'Models averaged', 'How they are combined',
                                     'Back-test typical error — our forecast vs the no-change guess', 'Back-test direction right (our forecast)',
                                     'Best single model in back-tests', f'{cv_best}: when training stopped (chosen on validation data)', f'{cv_best} settings'],
                            'Value': ['Average of the models', ', '.join(members) or '—', 'plain average of their forecast % changes (equal weights)',
                                      f"{status.get('combined_cv_rmse_ret') or float('nan'):.5f} vs {status.get('cv_rmse_ret_naive', float('nan')):.5f}",
                                      pct(status.get('combined_cv_dir_acc')),
                                      f"{cv_best or '—'} (lowest average error in the back-tests)", stop,
                                      ', '.join(f"{k} = {v}" for k, v in (status.get('params') or {}).items())]})
    st.dataframe(details, width='stretch', hide_index=True)
    st.caption("Full technical detail: `docs/FYP_Technical_Report.pdf`.")

st.markdown(f"<div class='footer'>{DISCLAIMER}</div>", unsafe_allow_html=True)
