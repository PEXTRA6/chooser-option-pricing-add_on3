# Chooser Option Pricing Dashboard

Streamlit dashboard for the JPMorgan Chooser Option Pricing project.

## Final interface

The app is organized into four focused pages:

- **Overview** — concise project positioning.
- **Live Pricing** — selectable market history, BSM theoretical price, frozen Residual XGBoost correction, headline changes, and market-condition charts.
- **Sensitivity** — frozen Week 7 volatility and interest-rate stress results.
- **Model Performance** — fixed out-of-sample validation and SHAP interpretation.

## Time-range presets

| UI range | Data period | Observation interval |
|---|---|---|
| 5 Years | 5y | 1d |
| 1 Year | 1y | 1d |
| 6 Months | 6mo | 1d |
| 1 Month | 1mo | 1d |
| 5 Days | 5d | 1h |
| 1 Day | 1d | 1h |

Longer horizons use daily observations because the model's rolling-volatility and momentum features are daily-oriented. Recent views use hourly market states for monitoring.

## Live Pricing display

Headline cards show:

- JPM — 3 decimals + change
- VIX — 3 decimals + change
- 10Y Treasury — 3 decimals + change
- BSM Price — 4 decimals + change
- ML-adjusted Price — 4 decimals + change

For daily presets, change is versus the previous trading day. For hourly presets, change is versus the previous hourly observation.

The page also displays:

- BSM vs ML-adjusted price trend
- ML correction and correction percentage
- ML correction through time
- VIX history
- 10Y Treasury history

Frozen-test RMSE metrics were intentionally removed from Live Pricing. They belong to historical model validation rather than current-market inference.

## Historical model validation

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| BSM Baseline | 0.349716 | 0.449818 | 0.999523 |
| Residual XGB λ=0.95 | 0.389886 | 0.506971 | 0.999394 |

These are fixed out-of-sample results and therefore do **not** update with current market data. BSM remained stronger on the frozen test set, so ML is presented as an experimental residual-correction layer.

## Repository structure

```text
.
├── streamlit_app.py
├── pricing_engine.py
├── requirements.txt
├── README.md
├── models/
│   ├── week8_config.json
│   ├── week7_residual_xgb.pkl   # add exported frozen Week 7 model
│   └── README.md
├── results/
├── assets/
├── notebooks/
└── data/
```

## Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Important model file note

The source package available when this repository was assembled did not contain `week7_residual_xgb.pkl`. The code expects it at `models/week7_residual_xgb.pkl`. Without that file, BSM and static research pages work, while ML-adjusted live output is disabled.

## Research boundary

The dashboard produces theoretical Chooser Option prices from public market inputs. It is not an observed Chooser Option transaction-price feed, and Yahoo Finance/yfinance is not an exchange-grade real-time data service.
