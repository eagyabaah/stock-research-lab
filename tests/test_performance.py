import pandas as pd
from stock_model.performance import append_signals, grade_ledger, performance_summary


def test_ledger_records_and_grades_long_signal():
    reports = [{"ticker":"XYZ","recommendation":"LONG","confidence":"HIGH","price":100,"long_score":85,"short_score":20,"strategy":"TEST","trade_plan":None}]
    rows = append_signals([], reports, "2026-01-02")
    idx = pd.to_datetime(["2026-01-02","2026-01-05","2026-01-06","2026-01-07","2026-01-08","2026-01-09"])
    hist = pd.DataFrame({"Close":[100,101,102,103,104,105]}, index=idx)
    spy = pd.DataFrame({"Close":[100,100,100,100,100,100]}, index=idx)
    rows = grade_ledger(rows, lambda _: hist, spy)
    assert rows[0]["outcomes"]["d5"]["win"] is True
    summary = performance_summary(rows, 5)
    assert summary["count"] == 1
    assert summary["win_rate"] == 100.0
