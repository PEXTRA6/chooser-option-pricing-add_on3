from pathlib import Path

import pandas as pd
import streamlit as st

from pricing_engine import TIME_PRESETS, build_live_series, cfg, get_new_york_time, load_model

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"
HISTORY = ROOT / "data" / "live_price_history.csv"

st.set_page_config(page_title="Chooser Option Pricing", page_icon="📈", layout="wide")
st.title("Chooser Option Pricing Dashboard")
st.caption("BSM benchmark + frozen Residual XGBoost + market-data integration")

overview, live_tab, sensitivity_tab, performance_tab = st.tabs(
    ["Overview", "Live Pricing", "Sensitivity", "Model Performance"]
)

with overview:
    st.subheader("Project overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dual Pricing", "BSM + ML")
    c2.metric("History", "Up to 5 Years")
    c3.metric("Sensitivity", "Vol + Rate")
    c4.metric("Validation", "MAE / RMSE / R²")
    st.info(
        "BSM remains the structural pricing benchmark. The frozen Residual XGBoost "
        "is used as an experimental residual-correction layer."
    )

with live_tab:
    st.subheader("Pricing & market history")
    st.caption("Select a horizon. Longer views use daily data; recent views use hourly data.")

    control1, control2, control3 = st.columns([2, 1, 1])
    with control1:
        time_range = st.selectbox("Time Range", list(TIME_PRESETS.keys()), index=0)
    preset = TIME_PRESETS[time_range]
    control2.metric("Data Frequency", preset["frequency"])
    control3.metric("New York Time", get_new_york_time().strftime("%H:%M %Z"))

    if st.button("Refresh Latest Data", type="primary"):
        st.cache_data.clear()

    try:
        s = build_live_series(period=preset["period"], interval=preset["interval"])
    except Exception as e:
        st.error(f"Pricing update failed: {e}")
        s = pd.DataFrame()

    if not s.empty:
        last = s.iloc[-1]
        prev = s.iloc[-2] if len(s) >= 2 else None

        def delta(col):
            if prev is None or pd.isna(last[col]) or pd.isna(prev[col]):
                return None
            return float(last[col] - prev[col])

        # One row per unique latest market timestamp for optional local audit history.
        HISTORY.parent.mkdir(parents=True, exist_ok=True)
        new_row = pd.DataFrame([last.to_dict()])
        old = pd.read_csv(HISTORY) if HISTORY.exists() else pd.DataFrame()
        out = pd.concat([old, new_row], ignore_index=True)
        if "Timestamp" in out.columns:
            out = out.drop_duplicates("Timestamp", keep="last")
        out.to_csv(HISTORY, index=False)

        jpm_d, vix_d, rate_d = delta("JPM"), delta("VIX"), delta("Treasury_10Y")
        bsm_d, ml_d = delta("BSM_Price"), delta("ML_Adjusted_Price")

        m1, m2, m3 = st.columns(3)
        m1.metric("JPM", f"${last.JPM:.3f}", delta=f"{jpm_d:+.3f}" if jpm_d is not None else None)
        m2.metric("VIX", f"{last.VIX:.3f}", delta=f"{vix_d:+.3f}" if vix_d is not None else None)
        m3.metric("10Y Treasury", f"{last.Treasury_10Y:.3f}%", delta=f"{rate_d:+.3f}" if rate_d is not None else None)

        p1, p2 = st.columns(2)
        p1.metric("BSM Price", f"${last.BSM_Price:.4f}", delta=f"{bsm_d:+.4f}" if bsm_d is not None else None)
        p2.metric(
            "ML-adjusted Price",
            f"${last.ML_Adjusted_Price:.4f}" if pd.notna(last.ML_Adjusted_Price) else "model file missing",
            delta=f"{ml_d:+.4f}" if ml_d is not None else None,
        )

        if pd.notna(last.ML_Correction):
            correction_pct = 100.0 * last.ML_Correction / last.BSM_Price if last.BSM_Price else float("nan")
            x1, x2 = st.columns(2)
            x1.metric("ML Correction", f"{last.ML_Correction:+.4f}")
            x2.metric("Correction %", f"{correction_pct:+.3f}%")

        st.markdown("### Price trend")
        price_cols = ["BSM_Price"]
        if s["ML_Adjusted_Price"].notna().any():
            price_cols.append("ML_Adjusted_Price")
        st.line_chart(s.set_index("Timestamp")[price_cols], height=400)

        if s["ML_Correction"].notna().any():
            st.markdown("### ML correction through time")
            st.line_chart(s.set_index("Timestamp")[["ML_Correction"]], height=220)

        st.markdown("### Market conditions")
        st.caption("VIX and 10Y Treasury use the same selected time range and observation frequency.")
        st.line_chart(s.set_index("Timestamp")[["VIX"]], height=220)
        st.line_chart(s.set_index("Timestamp")[["Treasury_10Y"]], height=220)

        observation_label = "previous trading day" if preset["interval"] == "1d" else "previous hourly observation"
        st.caption(f"Δ compares each headline metric with the {observation_label}.")
        st.caption(f"Latest available market timestamp (New York clock): {last.Timestamp}")
        st.caption("Research theoretical pricing; not an observed Chooser Option transaction price.")

        if load_model() is None:
            st.warning(
                "The repository does not currently contain `models/week7_residual_xgb.pkl`. "
                "Add the exported frozen Week 7 model to enable ML-adjusted price and correction output."
            )

with sensitivity_tab:
    st.subheader("Sensitivity analysis")
    st.caption("Frozen research outputs for volatility and interest-rate shocks.")
    scen = pd.read_csv(RESULTS / "sensitivity_scenarios.csv")
    st.dataframe(scen, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    vol = scen.loc[scen["Scenario"] == "Volatility +50%"].iloc[0]
    rate = scen.loc[scen["Scenario"] == "Rate +2pp"].iloc[0]
    combo = scen.loc[scen["Scenario"] == "Vol +50% & Rate +2pp"].iloc[0]
    c1.metric("Vol +50% — BSM Δ", f"{vol.BSM_Change_vs_Base:.4f}")
    c1.caption(f"ML-adjusted Δ {vol.ML_Change_vs_Base:.4f}")
    c2.metric("Rate +2pp — BSM Δ", f"{rate.BSM_Change_vs_Base:.4f}")
    c2.caption(f"ML-adjusted Δ {rate.ML_Change_vs_Base:.4f}")
    c3.metric("Combined — BSM Δ", f"{combo.BSM_Change_vs_Base:.4f}")
    c3.caption(f"ML-adjusted Δ {combo.ML_Change_vs_Base:.4f}")

    if (ASSETS / "volatility_sensitivity.png").exists():
        st.markdown("### Volatility sensitivity")
        st.image(str(ASSETS / "volatility_sensitivity.png"), use_container_width=True)
    if (ASSETS / "rate_sensitivity.png").exists():
        st.markdown("### Interest-rate sensitivity")
        st.image(str(ASSETS / "rate_sensitivity.png"), use_container_width=True)

with performance_tab:
    st.subheader("Historical model validation")
    st.caption(
        "Fixed out-of-sample test performance. These metrics are historical validation results "
        "and do not change with current market-data updates."
    )
    perf = pd.read_csv(RESULTS / "model_performance.csv")
    st.dataframe(perf, use_container_width=True, hide_index=True)

    st.info(
        "BSM remained the stronger frozen-test benchmark. The ML model is therefore deployed "
        "as an experimental residual correction rather than a replacement for BSM."
    )

    st.markdown("### SHAP feature importance")
    if (ASSETS / "shap_feature_importance.png").exists():
        st.image(str(ASSETS / "shap_feature_importance.png"), use_container_width=True)
    shap_df = pd.read_csv(RESULTS / "shap_importance_top15.csv")
    with st.expander("Show SHAP values"):
        st.dataframe(shap_df, use_container_width=True, hide_index=True)
