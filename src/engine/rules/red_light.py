"""Rule: vượt đèn đỏ.

Hỗ trợ 3 case nâng cao:
1. Rẽ [hướng] khi đèn đỏ nếu luật cho phép (`red_light_exceptions.allow_directions_on_red`,
   vd rẽ phải khi đỏ) HOẶC khi hướng đó không hề bị đèn nào quản lý (khai báo
   `traffic_lights[].apply_to.directions` không bao gồm hướng đó — vd đèn chỉ có mũi tên
   thẳng+trái, rẽ phải tự do — xem `TrafficLightEngine.direction_has_light()`, tự động suy ra,
   KHÔNG cần khai báo thêm `red_light_exceptions` riêng):
   - Lane CHỈ cho phép ĐÚNG 1 hướng (lane chuyên dụng 1 hướng): biết chắc hướng ngay lúc cắt
     vạch dừng — quyết định miễn trừ NGAY, không ghi nhận vi phạm nếu hướng đó được phép.
   - Lane cho phép NHIỀU hướng (vd làn chung cho thẳng+trái+phải, chỉ 1 đèn điều khiển chung):
     KHÔNG đủ thông tin xác định hướng thật ngay lúc cắt vạch — GHI NHẬN vi phạm NGAY (giống
     hành vi trước đây), rồi tiếp tục theo dõi xe tới khi cắt qua vạch `end_direction` (giống cơ
     chế `wrong_turn.py`) để biết hướng thật; nếu đúng hướng nằm trong danh sách miễn trừ HOẶC
     không bị đèn nào quản, RÚT LẠI vi phạm đã ghi (tái dùng pattern rút lại đã có ở
     `NoHelmetRule` — giữ evidence/log làm dấu vết audit trail, không xoá, chỉ đánh dấu
     `retracted`).
2. Nhiều đèn cho nhiều hướng trong cùng 1 lane — traffic_light.state_for_lane() hỗ trợ lọc
   theo hướng qua TrafficLightConfig.directions.
3. Ngoại lệ theo loại xe CHO TỪNG HƯỚNG bị đèn quản lý (`red_light_exceptions.exempt_classes`,
   dạng `{"<hướng>": ["<loại xe>", ...]}` — vd xe máy được đi tiếp khi đỏ lúc rẽ trái nhưng ô tô
   thì không). CHỈ áp dụng cho hướng CÓ đèn quản lý — hướng không bị đèn quản lý (case 1) đã tự
   do với MỌI loại xe rồi, không cần khai báo gì thêm ở đó. Cùng cơ chế hoãn-tới-khi-biết-hướng-
   thật như case 1 nếu lane nhiều hướng (chưa biết ngay xe đi hướng nào lúc cắt vạch dừng).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.engine.traffic_light import TrafficLightEngine
from src.engine.tracker import Track
from src.engine.zones import ZoneMap
from src.utils.geometry import polyline_crossed


@dataclass
class RedLightViolation:
    track_id: int
    lane_id: str
    frame_idx: int


@dataclass
class _PendingDirectionCheck:
    """Xe đã bị ghi nhận vượt đèn đỏ NGAY lúc cắt vạch (làn nhiều hướng, chưa biết hướng thật) —
    chờ xác nhận qua vạch end_direction để quyết định GIỮ hay RÚT LẠI. Chỉ giữ `lane_id`
    (không nhân bản `red_light_exceptions` ra đây) — tra lại lane khi cần ở thời điểm xác nhận,
    tránh 2 nơi có thể lệch nhau nếu sau này thêm trường mới vào exceptions."""
    lane_id: str


class RedLightRule:
    def __init__(self, zone_map: ZoneMap, traffic_lights: TrafficLightEngine):
        self.zone_map = zone_map
        self.traffic_lights = traffic_lights
        self._prev_points: dict[int, tuple[float, float]] = {}
        self._reported: set[int] = set()
        self._pending: dict[int, _PendingDirectionCheck] = {}
        # Tập hợp MỌI giá trị "direction" thực sự dùng trong config này (lấy từ chính các vạch
        # end_direction đã vẽ) — dùng để suy ra "hướng duy nhất lane cho phép" mà KHÔNG hardcode
        # tên hướng cụ thể ("straight"/"left"/"right" hay "ST"/"L"/"R" tuỳ config đặt tên khác
        # nhau, xem _single_allowed_direction()). Cùng cách WrongTurnRule đã làm.
        self._possible_directions = {
            line.direction for line in zone_map.lines_by_type("end_direction") if line.direction
        }

    def _single_allowed_direction(self, lane) -> str | None:
        """Hướng DUY NHẤT lane cho phép, hoặc None nếu lane cho phép 0/2+/mọi hướng — tính đúng
        qua `Zone.is_direction_allowed()` (tôn trọng cả `direction_mode` allow/deny), KHÔNG đọc
        thẳng `lane.directions` (field đó là danh sách ĐƯỢC PHÉP hoặc BỊ CẤM tuỳ `direction_mode`,
        đọc thẳng như hướng "được phép" sẽ SAI khi mode="deny")."""
        allowed = [d for d in self._possible_directions if lane.is_direction_allowed(d)]
        return allowed[0] if len(allowed) == 1 else None

    def update(self, tracks: list[Track], frame_idx: int) -> tuple[list[RedLightViolation], list[tuple[int, str]]]:
        """Trả về (vi phạm MỚI ghi nhận, danh sách (track_id, lý_do) vừa được RÚT LẠI) — lý do là
        1 trong 3 khoá cố định để lớp gọi (pipeline.py) map sang text hiển thị: "allow_on_red"
        (hướng nằm trong danh sách luật cho phép rẽ khi đỏ), "not_governed" (hướng đó hoá ra
        không bị đèn nào quản lý), hoặc "class_exempt" (loại xe được miễn trừ riêng cho hướng đó)."""
        violations: list[RedLightViolation] = []
        retracted: list[tuple[int, str]] = []

        for t in tracks:
            point = self.zone_map.anchor_point(t.bbox)
            prev = self._prev_points.get(t.track_id)
            self._prev_points[t.track_id] = point

            # (A) Xe đang chờ xác nhận hướng thật (đã ghi nhận vi phạm tạm thời) — kiểm tra có
            # vừa cắt qua vạch end_direction nào không, BẤT KỂ track có đang bị coi là _reported
            # hay không (chính là track đó, cố ý không bị chặn bởi check _reported bên dưới).
            pending = self._pending.get(t.track_id)
            if pending is not None and prev is not None:
                for line in self.zone_map.lines_by_type("end_direction"):
                    if not line.direction or not polyline_crossed(prev, point, line.points):
                        continue
                    pending_lane = self.zone_map.zones.get(pending.lane_id)
                    exceptions = (pending_lane.red_light_exceptions or {}) if pending_lane else {}
                    # Rút lại nếu (a) hướng xác nhận được nằm trong danh sách miễn trừ luật cho
                    # phép (khai báo tay), HOẶC (b) hướng đó không hề bị đèn nào quản lý (câu hỏi
                    # CẤU TRÚC, không phụ thuộc màu đèn hiện tại — tự động suy ra từ
                    # apply_to.directions, không cần khai báo tay), HOẶC (c) loại xe này được
                    # miễn trừ riêng cho đúng hướng vừa xác nhận (exempt_classes theo hướng).
                    exempt_by_law = line.direction in exceptions.get("allow_directions_on_red", [])
                    not_governed = not self.traffic_lights.direction_has_light(pending.lane_id, line.direction)
                    class_exempt = t.cls_name in exceptions.get("exempt_classes", {}).get(line.direction, [])
                    if exempt_by_law or not_governed or class_exempt:
                        self._reported.discard(t.track_id)
                        reason = "allow_on_red" if exempt_by_law else ("not_governed" if not_governed else "class_exempt")
                        retracted.append((t.track_id, reason))
                    del self._pending[t.track_id]
                    break

            if prev is None or t.track_id in self._reported:
                continue

            # Lấy lane trực tiếp từ CHÍNH vạch dừng bị cắt (stop_line.lane_id), KHÔNG dùng
            # find_lane(bbox hiện tại) — vì polygon lane chỉ vẽ tới đúng vạch dừng, không lấn
            # qua bên kia, nên ngay khi xe vừa cắt qua (đã ở "phía bên kia") thì điểm neo hiện
            # tại tự nhiên rơi ra ngoài polygon lane, find_lane() sẽ luôn trả None đúng lúc
            # cần nhất -> rule không bao giờ bắt được vi phạm thật (bug đã phát hiện 2026-07-08).
            for stop_line in self.zone_map.lines_by_type("stop_line"):
                if not stop_line.lane_id or len(stop_line.points) < 2:
                    continue
                if not polyline_crossed(prev, point, stop_line.points):
                    continue  # chưa cắt qua vạch dừng này ở frame này

                lane = self.zone_map.zones.get(stop_line.lane_id)
                if lane is None:
                    break

                # Lane chỉ cho 1 hướng duy nhất -> đủ tin cậy để suy ra hướng thật ngay tại vạch
                # dừng (tính đúng qua is_direction_allowed, xem _single_allowed_direction()).
                single_direction = self._single_allowed_direction(lane)
                # state_for_lane() đã tự trả UNKNOWN nếu không đèn nào quản `single_direction`
                # (xem TrafficLightEngine.state_for_lane) -> case "hướng tự do, không đèn quản"
                # của lane 1-hướng-duy-nhất tự động được miễn trừ ở đây, không cần code thêm.
                state = self.traffic_lights.state_for_lane(lane.id, direction=single_direction)
                if state != "red":
                    break

                exceptions = lane.red_light_exceptions or {}
                allow_on_red = exceptions.get("allow_directions_on_red", [])
                # exempt_classes: dict {"<hướng>": ["<loại xe>", ...]} — CHỈ có ý nghĩa cho hướng
                # CÓ đèn quản lý (hướng không bị quản lý đã tự do với mọi loại xe từ case 1, xem
                # docstring module). Case 3 cũ (list phẳng áp dụng cả lane, không phân biệt
                # hướng) đã bỏ — thống nhất về 1 dạng duy nhất theo hướng.
                exempt_classes_by_direction = exceptions.get("exempt_classes", {})

                # Case 1a + 3 (lane 1 hướng riêng): biết chắc hướng NGAY, miễn trừ luôn nếu khớp
                # luật HOẶC khớp loại xe miễn trừ cho đúng hướng đó — không cần theo dõi thêm gì
                # (khác case 1b bên dưới, lane nhiều hướng, chưa biết hướng thật ngay lúc này).
                if single_direction is not None:
                    if single_direction in allow_on_red:
                        break
                    if t.cls_name in exempt_classes_by_direction.get(single_direction, []):
                        break

                self._reported.add(t.track_id)
                violations.append(RedLightViolation(track_id=t.track_id, lane_id=lane.id, frame_idx=frame_idx))

                # Case 1b: lane nhiều hướng — CHƯA chắc đây có phải vi phạm thật hay không (chưa
                # biết hướng thật lúc này, kể cả loại xe có được miễn trừ hay không cũng phụ
                # thuộc hướng) — ghi nhận tạm (như trên) nhưng LUÔN đưa vào hàng chờ xác nhận qua
                # end_direction (xem nhánh (A) ở trên xử lý đủ cả 3 lý do rút lại).
                if single_direction is None:
                    self._pending[t.track_id] = _PendingDirectionCheck(lane_id=lane.id)
                break  # đã xử lý xong vạch dừng mà xe cắt qua, không cần xét vạch khác

        return violations, retracted

    def remove(self, track_ids: set[int]) -> None:
        for tid in track_ids:
            self._prev_points.pop(tid, None)
            self._reported.discard(tid)
            self._pending.pop(tid, None)
