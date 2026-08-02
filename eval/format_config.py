"""Định dạng gọn 1 file config JSON — xem src/utils/config_format.py để biết chi tiết thuật toán
(dùng chung với src/web/state.py, tránh trùng lặp code).

Ví dụ:
    python eval/format_config.py config/videos/vid_test.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_format import format_compact

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
with open(path, "w", encoding="utf-8") as f:
    f.write(format_compact(data) + "\n")
print("formatted", path)
