"""Load & validate config JSON (7 block: meta/classes/zones/traffic_lights/lines/vectors/system).
Schema đầy đủ: xem docs/brainstorm-notes.md.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REQUIRED_BLOCKS = ["meta", "classes", "zones", "traffic_lights", "lines", "vectors", "system"]
VALID_ZONE_TYPES = {"road", "lane", "wrong_way_zone", "traffic_light_zone"}
VALID_LINE_TYPES = {"stop_line", "end_direction"}
# meta.video_id: mã ngắn NGƯỜI DÙNG TỰ ĐẶT, dùng làm tiền tố track ID (vd "VD1_57") để track ID
# duy nhất TUYỆT ĐỐI giữa các video (không chỉ trong phạm vi 1 video) — xem docs/nhat-ky-ky-thuat.md.
# Giới hạn ký tự an toàn cho tên file/CSV (không dấu cách, không ký tự đặc biệt).
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,12}$")


class ConfigError(ValueError):
    pass


def load_config(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    warnings = validate_config(data)
    for w in warnings:
        print(f"CANH BAO CONFIG ({path}): {w}")
    return data


def validate_config(cfg: dict) -> list[str]:
    """Kiểm tra hợp lệ — báo lỗi CHẶN (`ConfigError`) cho id/tham chiếu sai, và trả về danh sách
    CẢNH BÁO KHÔNG CHẶN (vd lane thiếu đèn tín hiệu) để người gọi tự quyết định hiển thị/log lại,
    không ném ngoại lệ vì đây có thể là chủ đích hợp lệ (lane rẽ tự do không chịu đèn nào)."""
    missing = [b for b in REQUIRED_BLOCKS if b not in cfg]
    if missing:
        raise ConfigError(f"Config thiếu block bắt buộc: {missing}")

    video_id = cfg.get("meta", {}).get("video_id")
    if not video_id:
        raise ConfigError(
            "meta.video_id là bắt buộc — đặt 1 mã ngắn DUY NHẤT cho video này (vd 'VD1'), dùng làm "
            "tiền tố track ID để không nhầm lẫn giữa các video khác nhau (vd track ID 'VD1_57')."
        )
    if not VIDEO_ID_PATTERN.match(video_id):
        raise ConfigError(
            f"meta.video_id '{video_id}' không hợp lệ — chỉ dùng chữ/số/'_'/'-', tối đa 12 ký tự "
            f"(ngắn gọn vì sẽ ghép vào MỌI track ID của video này)."
        )

    zone_ids: set[str] = set()
    lane_ids: set[str] = set()
    for zone in cfg["zones"]:
        zid = zone.get("id")
        ztype = zone.get("type")
        if not zid or not ztype:
            raise ConfigError(f"Zone thiếu 'id' hoặc 'type': {zone}")
        if ztype not in VALID_ZONE_TYPES:
            raise ConfigError(f"Zone type không hợp lệ '{ztype}' (id={zid})")
        if zid in zone_ids:
            raise ConfigError(f"Zone id trùng lặp: {zid}")
        zone_ids.add(zid)
        # "lane_ids" o day thuc chat la "cac zone dung duoc lam lane" — ngoai type=lane con nhan
        # ca type=road (thuong role=analysis, "vung xuat phat"): video duong 1 lan khong can chia
        # lane rieng, dung thang vung xuat phat lam "lane" ngam cho stop_line/traffic_lights —
        # da xac nhan qua doc code: RedLightRule/TrafficLightEngine.state_for_lane() chi tra cuu
        # theo ID, KHONG kiem tra zone.type=="lane" o dau ca, chi validate_config truoc day tu
        # dat rang buoc chat hon can thiet. CHI ap dung cho stop_line/traffic_lights (duoi day) —
        # find_lane() (dung boi WrongLaneRule/WrongWayRule/WrongTurnRule) van chi nhan type=lane
        # (khong doi — nhung zone nay khong the tra ve tu do neu ko phai type=lane, xem zones.py).
        if ztype in ("lane", "road"):
            lane_ids.add(zid)

    line_ids: set[str] = set()
    for line in cfg["lines"]:
        lid = line.get("id")
        ltype = line.get("type")
        if not lid or not ltype:
            raise ConfigError(f"Line thiếu 'id' hoặc 'type': {line}")
        if ltype not in VALID_LINE_TYPES:
            raise ConfigError(f"Line type không hợp lệ '{ltype}' (id={lid})")
        if lid in line_ids:
            raise ConfigError(f"Line id trùng lặp: {lid}")
        line_ids.add(lid)
        if len(line.get("points", [])) < 2:
            raise ConfigError(f"Line '{lid}' cần ít nhất 2 điểm")
        if ltype == "stop_line":
            lane_id = line.get("lane_id")
            if lane_id and lane_id not in lane_ids:
                raise ConfigError(f"stop_line '{lid}' tham chiếu lane_id không tồn tại: {lane_id}")

    for tl in cfg["traffic_lights"]:
        tlz_id = tl.get("traffic_light_zone_id")
        if tlz_id and tlz_id not in zone_ids:
            raise ConfigError(f"traffic_light '{tl.get('id')}' tham chiếu traffic_light_zone_id không tồn tại: {tlz_id}")
        for lid in tl.get("apply_to", {}).get("lane_ids", []):
            if lid not in lane_ids:
                raise ConfigError(f"traffic_light '{tl.get('id')}' tham chiếu lane_id không tồn tại: {lid}")

    for vec in cfg["vectors"]:
        if "start" not in vec or "end" not in vec:
            raise ConfigError(f"Vector thiếu 'start'/'end': {vec}")

    # Cảnh báo (KHÔNG chặn): lane không được đèn tín hiệu nào áp dụng, và không đánh dấu tường
    # minh là được miễn — CHỈ kiểm tra khi config CÓ khai báo đèn tín hiệu (xác nhận đây là nút
    # giao có đèn thật, không phải đoạn đường thường không nút giao). Không phải lane nào cũng
    # bắt buộc có đèn (vd lane rẽ tự do) — nếu đúng vậy, khai báo "traffic_light_exempt": true
    # trong zones.<lane>.rules để tắt cảnh báo này 1 cách chủ động thay vì im lặng bỏ qua.
    # Đã từng là bug thật trong chính dự án này (lane L4 của vid_test.json từng thiếu đèn áp
    # dụng, chỉ phát hiện được qua rà soát thủ công — xem CLAUDE.md/nhat-ky-ky-thuat.md) —
    # validation này lẽ ra đã bắt được lỗi đó tự động.
    warnings: list[str] = []
    if cfg["traffic_lights"]:
        covered_lane_ids: set[str] = set()
        for tl in cfg["traffic_lights"]:
            covered_lane_ids.update(tl.get("apply_to", {}).get("lane_ids", []))
        for zone in cfg["zones"]:
            if zone.get("type") != "lane":
                continue
            zid = zone["id"]
            if zid in covered_lane_ids:
                continue
            exempt = zone.get("rules", {}).get("traffic_light_exempt", False)
            if not exempt:
                warnings.append(
                    f"Lane '{zid}' không được đèn tín hiệu nào áp dụng (traffic_lights[].apply_to."
                    f"lane_ids) và không đánh dấu 'rules.traffic_light_exempt: true' — xác nhận lại "
                    f"đây có đúng là lane không cần đèn hay bị sót cấu hình (xem CONFIG_GUIDE.md)."
                )
    return warnings


def generate_id(prefix: str, existing_ids) -> str:
    """Tự sinh ID theo pattern (vd L1, TLZ1) — dùng cho Config Wizard, tránh gõ tay trùng/sai ID."""
    existing = set(existing_ids)
    i = 1
    while f"{prefix}{i}" in existing:
        i += 1
    return f"{prefix}{i}"
