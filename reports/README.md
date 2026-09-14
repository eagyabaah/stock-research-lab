# Generated closing reports

The GitHub Actions workflow writes `latest.json` and a dated JSON report into
this directory after the U.S. market close on weekdays. Do not edit generated
JSON files by hand.

The file `prediction_ledger.json` is the prospective audit trail for the scheduled US model. GitHub Actions appends timestamped LONG, SHORT, and NO TRADE decisions and forward-grades them at 1, 5, 10, and 20 trading-day horizons.
