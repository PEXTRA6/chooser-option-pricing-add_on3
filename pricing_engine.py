import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "models" / "week8_config.json"
MODEL_PATH = ROOT / "models" / "week7_residual_xgb.pkl"
cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
NY_TZ = "America/New_York"

TIME_PRESETS = {
    "5 Years": {"period": "5y", "interval": "1d", "frequency": "Daily"},
    "1 Year": {"period": "1y", "interval": "1d", "frequency": "Daily"},
    "6 Months": {"period": "6mo", "interval": "1d", "frequency": "Daily"},
    "1 Month": {"period": "1mo", "interval": "1d", "frequency": "Daily"},
    "5 Days": {"period": "5d", "interval": "1h", "frequency": "Hourly"},
    "1 Day": {"period": "1d", "interval": "1h", "frequency": "Hourly"},
}


def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def get_new_york_time():
    return pd.Timestamp.now(tz=NY_TZ)


def chooser_bsm(S, K, r, q, sigma, T1, T2):
    S, K, r, q, sigma, T1, T2 = map(float, (S, K, r, q, sigma, T1, T2))
    if S <= 0 or K <= 0 or sigma <= 0 or T1 <= 0 or T2 <= T1:
        return np.nan
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T2) / (sigma * np.sqrt(T2))
    d2 = d1 - sigma * np.sqrt(T2)
    call = S * np.exp(-q * T2) * norm.cdf(d1) - K * np.exp(-r * T2) * norm.cdf(d2)
    k_star = K * np.exp(-(r - q) * (T2 - T1))
    e1 = (np.log(S / k_star) + (r - q + 0.5 * sigma**2) * T1) / (sigma * np.sqrt(T1))
    e2 = e1 - sigma * np.sqrt(T1)
    put = k_star * np.exp(-r * T1) * norm.cdf(-e2) - S * np.exp(-q * T1) * norm.cdf(-e1)
    return float(call + put)


def _close(ticker, period, interval):
    raw = yf.download(ticker, period=period, interval=interval, auto_adjust=True,
                      progress=False, threads=False, prepost=False)
    if raw.empty:
        raise RuntimeError(f"No market data returned for {ticker}")
    s = raw["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce").dropna()


def _to_ny_index(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        # Daily Yahoo timestamps are date labels; hourly labels are normally tz-aware.
        df.index = df.index.tz_localize(NY_TZ)
    else:
        df.index = df.index.tz_convert(NY_TZ)
    return df


def _series_to_ny(s):
    """Normalize a Yahoo series to a sorted America/New_York DatetimeIndex."""
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    if s.index.tz is None:
        s.index = s.index.tz_localize(NY_TZ)
    else:
        s.index = s.index.tz_convert(NY_TZ)
    return s[~s.index.duplicated(keep="last")].sort_index()


def _aligned_market(period, interval):
    """Align market inputs without requiring identical Yahoo timestamps.

    JPM is the clock. VIX and Treasury are matched to the most recent observation
    available at or before each JPM timestamp. This is especially important for
    hourly Yahoo data, where ^TNX timestamps can differ from JPM/^VIX.
    """
    jpm = _series_to_ny(_close("JPM", period, interval).rename("Close"))
    vix = _series_to_ny(_close("^VIX", period, interval).rename("VIX"))
    tnx = _series_to_ny(_close("^TNX", period, interval).rename("Treasury_10Y"))

    # Regular US equity session for intraday views.
    if interval != "1d":
        jpm = jpm[jpm.index.weekday < 5].between_time("09:30", "16:00")
        vix = vix[vix.index.weekday < 5].between_time("09:30", "16:00")
        tnx = tnx[tnx.index.weekday < 5].between_time("09:30", "16:00")
    else:
        jpm = jpm[jpm.index.weekday < 5]
        vix = vix[vix.index.weekday < 5]
        tnx = tnx[tnx.index.weekday < 5]

    if jpm.empty:
        return pd.DataFrame(columns=["Close", "VIX", "Treasury_10Y"])

    market = jpm.to_frame().sort_index()
    market = pd.merge_asof(
        market, vix.to_frame().sort_index(),
        left_index=True, right_index=True, direction="backward"
    )
    market = pd.merge_asof(
        market, tnx.to_frame().sort_index(),
        left_index=True, right_index=True, direction="backward"
    )

    # Forward-fill only within the returned series; never use future observations.
    market[["VIX", "Treasury_10Y"]] = market[["VIX", "Treasury_10Y"]].ffill()
    return market.dropna()


def download_market(period="5y", interval="1d"):
    """Return daily feature context plus the selected display series."""
    # Six years gives the 5Y display enough lookback for 60D rolling features.
    daily = _aligned_market("6y", "1d")
    display = _aligned_market(period, interval)

    # A 1-day hourly Yahoo request can be empty before the US session opens, on
    # weekends, or when one intraday feed is delayed. Fall back to a 5-day pull
    # and show the most recent regular trading session instead of failing.
    if display.empty and interval != "1d":
        fallback = _aligned_market("5d", interval)
        if not fallback.empty:
            if period == "1d":
                last_day = fallback.index[-1].date()
                display = fallback[fallback.index.date == last_day]
            else:
                display = fallback

    if len(daily) < 65:
        raise RuntimeError("Insufficient daily market history for rolling features.")
    if display.empty:
        raise RuntimeError("No market observations are currently available for this range.")
    return daily, display


def _feature_context(daily, timestamp, S, VIX, rate):
    """Create point-in-time daily context without using future observations."""
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize(NY_TZ)
    else:
        ts = ts.tz_convert(NY_TZ)

    day = ts.normalize()
    hist = daily[daily.index.normalize() < day].copy()
    current = pd.DataFrame(
        {"Close": [S], "VIX": [VIX], "Treasury_10Y": [rate]}, index=[ts]
    )
    return pd.concat([hist.tail(260), current])


def live_features(daily, S, VIX, rate, timestamp):
    K = float(cfg["K"])
    x = _feature_context(daily, timestamp, S, VIX, rate)
    if len(x) < 65:
        return None

    c, v, r = x["Close"], x["VIX"], x["Treasury_10Y"]
    ret = c.pct_change()
    lr = np.log(c / c.shift(1))
    vr = v.pct_change()

    f = {
        "Close": float(c.iloc[-1]),
        "Moneyness": float(c.iloc[-1] / K),
        "Log_Moneyness": float(np.log(c.iloc[-1] / K)),
        "Daily_Return": float(ret.iloc[-1]),
        "Log_Return": float(lr.iloc[-1]),
        "Rolling_Vol_5D": float(ret.rolling(5).std().iloc[-1] * np.sqrt(252)),
        "Rolling_Vol_20D": float(ret.rolling(20).std().iloc[-1] * np.sqrt(252)),
        "Rolling_Vol_60D": float(ret.rolling(60).std().iloc[-1] * np.sqrt(252)),
        "VIX": float(v.iloc[-1]),
        "VIX_Return": float(vr.iloc[-1]),
        "Treasury_10Y": float(r.iloc[-1]),
        "Rate_Momentum_5D": float(r.iloc[-1] - r.iloc[-6]),
        "Rate_Momentum_20D": float(r.iloc[-1] - r.iloc[-21]),
        "Rate_Pct_Change_5D": float(r.pct_change(5).iloc[-1]),
        "Rate_Pct_Change_20D": float(r.pct_change(20).iloc[-1]),
        "VIX_JPM_Correlation_20D": float(ret.rolling(20).corr(vr).iloc[-1]),
        "Price_Momentum_5D": float(c.pct_change(5).iloc[-1]),
        "Price_Momentum_20D": float(c.pct_change(20).iloc[-1]),
    }
    f["VIX_Decimal"] = f["VIX"] / 100.0
    f["Volatility_Spread"] = f["VIX_Decimal"] - f["Rolling_Vol_20D"]
    f["Volatility_Ratio"] = f["VIX_Decimal"] / max(f["Rolling_Vol_20D"], 1e-8)

    # Point-in-time VIX-derived sentiment proxy.
    vix_hist = v.dropna()
    lo, hi = float(vix_hist.quantile(0.01)), float(vix_hist.quantile(0.99))
    f["Market_Sentiment_Index"] = float(np.clip(1 - (f["VIX"] - lo) / max(hi - lo, 1e-8), 0, 1))
    f["Sentiment_x_Volatility"] = f["Market_Sentiment_Index"] * f["Rolling_Vol_20D"]
    f["BSM_Analytical_Price"] = chooser_bsm(
        f["Close"], K, f["Treasury_10Y"] / 100.0, float(cfg["Q"]),
        max(f["Rolling_Vol_60D"], 1e-4), float(cfg["T1"]), float(cfg["T2"])
    )
    return f


def build_live_series(period="5y", interval="1d"):
    """Build the selected daily/hourly theoretical pricing history."""
    model = load_model()
    daily, display = download_market(period, interval)
    rows = []

    for ts, row in display.iterrows():
        f = live_features(daily, float(row["Close"]), float(row["VIX"]),
                          float(row["Treasury_10Y"]), ts)
        if f is None:
            continue

        resid = corr = ml = np.nan
        if model is not None:
            features = cfg["MODEL_FEATURES"]
            missing = [name for name in features if name not in f or pd.isna(f[name])]
            if not missing:
                X = pd.DataFrame([{name: f[name] for name in features}])
                resid = float(model.predict(X)[0])
                corr = float(cfg["BEST_LAMBDA"]) * resid
                ml = float(f["BSM_Analytical_Price"] + corr)

        rows.append({
            "Timestamp": ts,
            "JPM": f["Close"],
            "VIX": f["VIX"],
            "Treasury_10Y": f["Treasury_10Y"],
            "Rolling_Vol_60D": f["Rolling_Vol_60D"],
            "Market_Sentiment_Index": f["Market_Sentiment_Index"],
            "BSM_Price": f["BSM_Analytical_Price"],
            "Predicted_Residual": resid,
            "ML_Correction": corr,
            "ML_Adjusted_Price": ml,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values("Timestamp").reset_index(drop=True)
    result["Timestamp"] = pd.to_datetime(result["Timestamp"])
    if result["Timestamp"].dt.tz is not None:
        result["Timestamp"] = result["Timestamp"].dt.tz_convert(NY_TZ).dt.tz_localize(None)
    return result
