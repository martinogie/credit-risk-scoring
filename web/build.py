"""Rebuild web/index.html from the latest training outputs in reports/."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
m = json.loads((root / "reports/metrics.json").read_text())
small = {k: m[k] for k in ["scorecard_oot", "gbm_challenger_oot", "score_psi", "data"]}
html = (root / "web/template.html").read_text()
html = html.replace("__SCORECARD__", (root / "reports/scorecard.json").read_text()).replace("__METRICS__", json.dumps(small))
(root / "web/index.html").write_text(html)
print("wrote web/index.html")
