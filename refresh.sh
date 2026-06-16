#!/bin/bash
python parse_logs.py
python generate_report.py
git add report.html
git commit -m "update results $(date '+%Y-%m-%d %H:%M')"
git push
echo "✅ Live at https://webdevsha.github.io/vendsafe-lab/report.html"
