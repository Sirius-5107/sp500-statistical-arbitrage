# S&P 500 Statistical Arbitrage Research

A research project testing whether same-sector S&P 500 equity pairs
selected using rolling correlation and bidirectional Engle–Granger
cointegration exhibit reliable out-of-sample mean reversion.

## Current status

The data pipeline, validation framework, candidate generation,
cointegration testing, multiple-testing correction, and out-of-sample
spread event study are complete.

The current universe uses present-day S&P 500 constituents and Yahoo
Finance data from 2000 onward. Consequently, the study is subject to
survivorship bias and should not be interpreted as a fully point-in-time
historical simulation.

## Preliminary findings

- 503 securities downloaded and validated
- 527,295 rolling candidate-pair snapshots
- 135,543 bidirectional cointegration tests
- Same-issuer share classes excluded
- 298 statistically qualified pair-months
- 168 out-of-sample divergence events
- 33.3% reached an absolute z-score below 1
- 23.2% reached an absolute z-score below 0.5
- 14.9% crossed the fitted spread mean

The preliminary results suggest that historical cointegration is
selective and unstable out of sample. A small number of structural
breakdowns produce substantial continued divergence.

## Reproducing the pipeline

Run the scripts in this order:

```bash
python scripts/download_constituents.py
python scripts/download_prices.py
python scripts/validate_data.py
python scripts/inspect_anomalies.py
python scripts/build_processed_data.py
python scripts/build_panels.py
python scripts/generate_candidates.py
python scripts/test_cointegration.py
python scripts/analyze_cointegration_sensitivity.py
python scripts/build_baseline_pairs.py
python scripts/run_spread_event_study.py

```

> **Checkpoint note:** The committed summary tables were produced during
> iterative development. A final clean pipeline run will be performed after
> the baseline backtest is frozen; exact intermediate counts may differ when
> regenerated with the latest same-issuer filtering logic.
