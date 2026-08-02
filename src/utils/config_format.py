"""Định dạng gọn JSON config — mảng số đơn giản (toạ độ, màu...) gộp 1 dòng, chỉ object/mảng
phức tạp mới xuống dòng riêng (khác `json.dumps(..., indent=2)` mặc định, vốn tách MỌI phần tử
lồng nhau xuống dòng riêng kể cả toạ độ đơn giản — rất khó đọc với config nhiều toạ độ polygon).
Dùng chung bởi `eval/format_config.py` (CLI) và `src/web/state.py` (lưu config sửa qua Web UI).
"""
from __future__ import annotations

import json


def _is_simple(v: object) -> bool:
    return isinstance(v, (int, float, str, bool)) or v is None


def _is_simple_list(v: object) -> bool:
    return isinstance(v, list) and all(_is_simple(x) for x in v)


def format_compact(obj: object, indent: int = 0, step: int = 2) -> str:
    pad = " " * indent
    pad_in = " " * (indent + step)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [f'{pad_in}"{k}": {format_compact(v, indent + step, step)}' for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, list):
        if not obj:
            return "[]"
        if _is_simple_list(obj):
            return "[" + ", ".join(json.dumps(x, ensure_ascii=False) for x in obj) + "]"
        if all(_is_simple_list(v) for v in obj):
            return "[" + ", ".join("[" + ", ".join(json.dumps(x, ensure_ascii=False) for x in v) + "]" for v in obj) + "]"
        items = [pad_in + format_compact(v, indent + step, step) for v in obj]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    return json.dumps(obj, ensure_ascii=False)
