"""Rebuild the demo page from the latest training outputs in reports/.

Writes web/index.html (page body) and docs/index.html (standalone page served by GitHub Pages).
"""
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
m = json.loads((root / "reports/metrics.json").read_text())
small = {k: m[k] for k in ["scorecard_oot", "gbm_challenger_oot", "score_psi", "data"]}
body = (root / "web/template.html").read_text()
body = body.replace("__SCORECARD__", (root / "reports/scorecard.json").read_text()).replace("__METRICS__", json.dumps(small))
(root / "web/index.html").write_text(body)

head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<meta name="description" content="Interactive consumer-loan credit scorecard: points, probability of default, decision band and adverse-action reasons.">\n'
        '<style>body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style>\n')
# Move the page's own <title>, font links and <style> into <head>; the rest goes in <body>.
split = body.index('<div class="wrap">')
page = head + body[:split] + "</head>\n<body>\n" + body[split:] + "\n</body>\n</html>\n"
(root / "docs").mkdir(exist_ok=True)
(root / "docs/index.html").write_text(page)
print("wrote web/index.html and docs/index.html")
