Finance Stock Prediction — End-to-end Example
===========================================

Overview
--------
This small project downloads historical stock prices (default: AAPL), performs feature engineering, trains a simple regression model to predict next-day returns, and creates visualizations summarizing results.

Quick start
-----------
1. Create a virtualenv and install deps:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the analysis (defaults to AAPL):

```powershell
python src\run_analysis.py --ticker AAPL --period 5y
```

Outputs
-------
- `data/` — downloaded CSV if network available
- `outputs/` — plots and saved model
- `src/run_analysis.py` — main pipeline

Files
-----
- [src/run_analysis.py](src/run_analysis.py)
- [requirements.txt](requirements.txt)
- [README.md](README.md)

Next steps
----------
- Swap `--ticker` to experiment with other tickers.
- Convert `src/run_analysis.py` to a Jupyter notebook for interactive exploration.
