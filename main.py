"""CLI: chạy hệ thống trên 1 video theo quy ước đặt tên.

Quy ước: video `data/input/<name>.<ext>` luôn đi kèm config `config/videos/<name>.json`,
và bên trong config, `meta.video_name` phải khớp đúng tên file video — không thì báo lỗi
rõ ràng thay vì chạy sai lặng lẽ.

Ví dụ:
    python main.py vid_test
    python main.py vid_test --show
    python main.py video_720p --max-frames 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.engine.config_schema import ConfigError, load_config
from src.engine.pipeline import Pipeline

INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")
CONFIG_DIR = Path("config/videos")


def resolve_video_path(name: str) -> Path:
    matches = sorted(INPUT_DIR.glob(f"{name}.*"))
    if not matches:
        raise FileNotFoundError(f"Không tìm thấy video '{name}' trong {INPUT_DIR}/")
    return matches[0]


def resolve_config_path(name: str) -> Path:
    path = CONFIG_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy config '{path}'. Mỗi video cần 1 file config cùng tên "
            f"(vd {INPUT_DIR}/{name}.mp4 <-> {CONFIG_DIR}/{name}.json)."
        )
    return path


def resolve_and_validate(name: str) -> tuple[Path, dict]:
    """Tìm video + config theo quy ước đặt tên, validate `meta.video_name` khớp đúng — dùng
    chung cho cả CLI (`main()` bên dưới) lẫn Web UI (`src/web/state.py`), tránh lặp logic."""
    video_path = resolve_video_path(name)
    config_path = resolve_config_path(name)
    config = load_config(str(config_path))

    actual_video_name = config.get("meta", {}).get("video_name")
    if actual_video_name != video_path.name:
        raise ConfigError(
            f"{config_path} có meta.video_name='{actual_video_name}' nhưng file video "
            f"thực tế là '{video_path.name}' — sửa lại cho khớp."
        )
    return video_path, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hệ thống phát hiện & giám sát vi phạm giao thông")
    parser.add_argument("name", help="Tên video, không kèm đuôi file — vd: vid_test")
    parser.add_argument("--model", default="models/pretrained/yolo11s.pt", help="Model YOLO dùng để detect/track")
    parser.add_argument("--show", action="store_true", help="Hiển thị cửa sổ preview (nhấn q để thoát)")
    parser.add_argument("--no-save", action="store_true", help="Không lưu video output (chỉ xem/đo FPS)")
    parser.add_argument("--max-frames", type=int, default=None, help="Giới hạn số frame xử lý (debug nhanh)")
    parser.add_argument(
        "--no-log", action="store_true",
        help="Không ghi log vi phạm/ảnh minh chứng (data/logs, data/evidence) — dùng khi debug lặp lại nhiều lần",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path, config = resolve_and_validate(args.name)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = None if args.no_save else str(OUTPUT_DIR / f"{args.name}.mp4")

    pipeline = Pipeline(config=config, model_path=args.model)
    stats = pipeline.run(
        video_path=str(video_path),
        output_path=output_path,
        show=args.show,
        max_frames=args.max_frames,
        log_name=None if args.no_log else args.name,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
