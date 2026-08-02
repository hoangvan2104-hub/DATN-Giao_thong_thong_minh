# Hướng dẫn thiết lập config cho 1 đoạn đường/camera mới

> File này giải thích cấu trúc file config JSON (`config/videos/<tên_video>.json`) để bạn có
> thể tự tạo/chỉnh config cho video mới mà không cần hỏi lại. Có ví dụ tình huống giao thông
> thật kèm theo từng phần. Xem thêm [docs/architecture.md](../docs/architecture.md) cho sơ đồ
> tổng thể, [docs/brainstorm-notes.md](../docs/brainstorm-notes.md) cho lý do thiết kế.

## Quy tắc đặt tên file

Video `data/input/<name>.<ext>` **luôn đi kèm** config `config/videos/<name>.json` — tên file
phải khớp nhau, và bên trong config, `meta.video_name` phải khớp đúng tên file video thật.
`main.py` validate chặt điều này, chạy `python main.py <name>` là đủ.

## Tổng quan 7 block bắt buộc

```json
{
  "meta": { },
  "classes": [ ],
  "zones": [ ],
  "traffic_lights": [ ],
  "lines": [ ],
  "vectors": [ ],
  "system": { }
}
```

Thiếu block nào (kể cả để mảng/object rỗng `[]`/`{}`) sẽ báo lỗi rõ ràng khi chạy — xem
`src/engine/config_schema.py`.

| Block | Ý nghĩa |
|---|---|
| `meta` | Tên video, mô tả, ngày tạo |
| `classes` | Danh sách lớp phương tiện model detect được (phải khớp tên class thật của model) |
| `zones` | Vùng đa giác: đường (`road`), làn (`lane`), vùng kiểm tra ngược chiều (`wrong_way_zone`), vùng đèn (`traffic_light_zone`) |
| `traffic_lights` | Cấu hình đọc màu đèn (detect/manual) + áp dụng cho lane/hướng nào |
| `lines` | Vạch dừng (`stop_line`) và vạch xác định hướng thoát (`end_direction`) |
| `vectors` | Hướng lưu thông đúng, dùng cho rule ngược chiều |
| `system` | Tham số kỹ thuật: tracker, detection, bật/tắt tính năng |

---

## Block `meta`

```json
{
  "video_name": "vid_test.mp4",
  "video_id": "VDT",
  "created_at": "2026-04-07",
  "description": "Nga tu co den giao thong",
  "recording_started_at": "2026-07-18T08:30:00"
}
```

- `video_name`: **bắt buộc**, phải khớp đúng tên file video thật (xem "Quy tắc đặt tên file").
- `video_id`: **bắt buộc**, mã ngắn TỰ ĐẶT, DUY NHẤT giữa mọi video trong hệ thống (chữ/số/`_`/`-`,
  tối đa 12 ký tự, vd `"VDT"`, `"CAM1"`). Dùng làm tiền tố ghép vào track_id khi ghi ra file
  (`events.csv/json`, `vehicle_summary.csv`, tên ảnh evidence — vd track_id thô `57` của video này
  ghi ra thành `VDT_57`) để track_id duy nhất TUYỆT ĐỐI giữa các video, không chỉ trong phạm vi 1
  video (track_id thô là số nội bộ của tracker, có thể trùng giữa các video khác nhau). Thiếu field
  này hoặc sai định dạng sẽ bị chặn ngay lúc load config (`ConfigError`). Sửa qua Web UI (trang chi
  tiết camera → Config) sẽ tự chặn nếu trùng với `video_id` của video khác đã có config.
- `created_at`, `description`: tự do, chỉ để ghi chú, không ảnh hưởng logic.
- `recording_started_at` (tuỳ chọn, ISO 8601 `YYYY-MM-DDTHH:MM:SS`): ngày giờ THẬT lúc camera bắt
  đầu quay video này — dùng để tính ngày giờ THẬT của từng vi phạm (`base + số giây trôi qua
  trong video`), hiện lên tên file ảnh minh chứng (`data/evidence/.../frame_..._<ngày giờ>.jpg`)
  và cột `violation_datetime` trong `data/logs/.../events.csv`. Chỉ dùng cho mục đích LỊCH SỬ/hồ
  sơ khi biết chắc thời điểm camera thật sự bắt đầu quay (vd nhập lại từ metadata gốc của file),
  không có ý nghĩa realtime. Không khai báo thì hệ thống lấy đúng THỜI ĐIỂM BẮT ĐẦU XỬ LÝ (giờ hệ
  thống lúc bấm chạy) làm mốc — trước đây dùng tạm mtime file video làm dự phòng nhưng đã đổi
  (2026-07-26, theo yêu cầu user) vì mtime phản ánh lúc file được ghi/copy vào máy (vd video demo
  nằm sẵn trong repo từ đầu dự án), không liên quan gì đến lúc thật sự chạy xử lý, dễ gây hiểu
  nhầm ngày vi phạm là ngày cũ dù vừa mới chạy.

---

## Block `zones` — chi tiết loại `lane`

Mỗi `lane` (làn đường) có `rules` gồm 3 phần **độc lập với nhau**, mỗi phần được 1 rule engine
riêng đọc:

```json
{
  "id": "L1",
  "type": "lane",
  "name": "lan_re_trai",
  "polygon": [[x1,y1], [x2,y2], ...],
  "rules": {
    "direction_mode": {
      "mode": "allow",
      "directions": ["left"]
    },
    "distribution_classes": {
      "mode": "deny",
      "classes": ["truck", "bus"]
    },
    "red_light_exceptions": {
      "exempt_classes": ["motorcycle"],
      "allow_directions_on_red": ["right"]
    },
    "traffic_light_exempt": false
  }
}
```

### 1. `direction_mode` — xe được rẽ hướng nào từ làn này (dùng bởi `rules/wrong_turn.py`)

- `mode`: `"allow"` (chỉ cho các hướng liệt kê) hoặc `"deny"` (cấm các hướng liệt kê, còn lại đều được).
- `directions`: danh sách hướng (`"left"`/`"straight"`/`"right"`, khớp `direction` khai báo ở
  các vạch `end_direction` trong block `lines`), hoặc chuỗi `"all"` (không giới hạn hướng).

**Ví dụ thật**: làn trong cùng của 1 ngã tư 4 hướng thường CHỈ cho rẽ trái — khai
`{"mode": "allow", "directions": ["left"]}`. Nếu xe từ làn này lại đi thẳng hoặc rẽ phải →
`rules/wrong_turn.py` phát hiện là **sai hướng rẽ**.

### 2. `distribution_classes` — loại phương tiện được đi vào làn này (dùng bởi `rules/wrong_lane.py`)

- `mode`: `"allow"` hoặc `"deny"`, cùng cơ chế như trên.
- `classes`: danh sách tên class (phải khớp với `classes` khai báo ở đầu file), hoặc `"all"`.

**Ví dụ thật**: làn hỗn hợp xe máy + ô tô con nhưng cấm xe tải/xe khách lớn (đường nhỏ, cầu
yếu tải trọng...) → `{"mode": "deny", "classes": ["truck", "bus"]}`. Xe tải cố tình đi vào →
**sai làn theo loại phương tiện**. Vi phạm này **không tính ngay khi vừa chạm vào làn** (tránh
báo sai khi xe chỉ lướt qua để đổi làn) — chỉ tính sau khi xe đã ở trong làn cấm đủ lâu
(mặc định 15 frame hoặc 20px, xem `WrongLaneRule`).

### 3. Không khai báo thì mặc định là gì?

Nếu 1 lane **không khai báo** `direction_mode` hoặc `distribution_classes` (hoặc khai báo
thiếu trường), hệ thống tự mặc định `{"mode": "allow", "directions"/"classes": "all"}` —
tức là **không giới hạn gì cả**, mọi hướng/mọi loại xe đều hợp lệ, không có gì bị coi là vi
phạm. Điều này an toàn (không tự ý tạo vi phạm giả khi bạn chưa kịp cấu hình đầy đủ), nhưng
nếu bạn MUỐN áp luật cho 1 lane thì bắt buộc phải khai rõ — hệ thống không tự đoán.

---

## Đèn tín hiệu — quan hệ với `direction_mode` và `red_light_exceptions`

Đây là phần nhiều bạn hay nhầm nên giải thích kỹ:

### `direction_mode`/`distribution_classes` **KHÔNG** trực tiếp liên kết với đèn đỏ

3 phần trong `rules` (direction_mode, distribution_classes, red_light_exceptions) được
**3 rule engine khác nhau đọc độc lập** (`wrong_turn.py`, `wrong_lane.py`, `red_light.py`).
Chúng cùng nằm trong 1 lane vì cùng mô tả "luật của lane này", nhưng không gọi lẫn nhau.

**Điểm kết nối duy nhất**: `red_light.py` cần biết xe đang **rẽ hướng nào** để tra đúng trạng
thái đèn cho hướng đó (xem mục dưới) — và nó lấy thông tin này từ CHÍNH `direction_mode` của
lane (chỉ suy luận được khi lane CHỈ cho đúng 1 hướng — xem phần "Giới hạn kỹ thuật" cuối file).

### Trường hợp lane không gán đèn tín hiệu nào (`traffic_lights` không có `apply_to.lane_ids` chứa lane đó)

`rules/red_light.py` gọi `traffic_lights.state_for_lane(lane_id)` — nếu lane không được đèn
nào áp dụng, hàm này trả về `"unknown"`. Rule chỉ báo vi phạm khi trạng thái đúng bằng
`"red"` — `"unknown"` **không bao giờ** bị tính là vi phạm. Nói cách khác: **lane không có
đèn tín hiệu thì mặc nhiên không bao giờ bị bắt lỗi vượt đèn đỏ** (hợp lý — lấy gì mà vượt).
Phù hợp cho các lane không có đèn thật (đường nhánh có biển "nhường đường" thay vì đèn, hoặc
đèn ở xa/khuất không vẽ vùng `traffic_light_zone` được).

**Cảnh báo tự động khi lane "mồ côi" đèn ở nút giao CÓ đèn**: nếu config CÓ khai báo ít nhất 1
đèn tín hiệu (`traffic_lights` không rỗng — xác nhận đây là nút giao có đèn thật, không phải
đoạn đường thường), `validate_config()` sẽ quét MỌI lane và cảnh báo (không chặn lưu/chạy) nếu
lane nào không được đèn nào áp dụng VÀ không khai báo `"rules": {"traffic_light_exempt": true}`
— vì trên thực tế đây thường là **thiếu sót cấu hình** (quên gán đèn cho 1 lane) chứ không phải
chủ đích, và lỗi loại này đã từng xảy ra thật trong dự án (1 lane bị bỏ sót đèn, chỉ phát hiện
qua rà thủ công). Nếu lane đó **thực sự** không cần đèn (lane rẽ tự do, đường nhánh có biển
nhường đường...), khai báo tường minh `traffic_light_exempt: true` để tắt cảnh báo — hệ thống
không tự đoán, phải người cấu hình xác nhận rõ ràng.

### Đèn tín hiệu điều phối theo LANE, và có thể điều phối theo HƯỚNG trong cùng 1 lane

Thực tế nhiều ngã tư có 1 lane rộng nhưng có 2 đèn: 1 đèn mũi tên riêng cho rẽ trái, 1 đèn
tròn cho đi thẳng — cùng 1 lane nhưng 2 hướng có thể có trạng thái đèn KHÁC NHAU tại cùng 1
thời điểm (đèn rẽ trái đỏ trong khi đèn đi thẳng xanh). Khai báo bằng cách thêm `directions`
vào `apply_to` của từng đèn trong block `traffic_lights`:

```json
{
  "id": "TL_left",
  "mode": "detect",
  "traffic_light_zone_id": "TLZ_1",
  "apply_to": { "lane_ids": ["L1"], "directions": ["left"] }
},
{
  "id": "TL_straight",
  "mode": "detect",
  "traffic_light_zone_id": "TLZ_2",
  "apply_to": { "lane_ids": ["L1"], "directions": ["straight"] }
}
```

Nếu không khai `directions` (bỏ trống) thì đèn đó áp dụng cho MỌI hướng của lane — đúng
trường hợp phổ biến nhất (1 lane 1 đèn).

### Rẽ hướng nào đó được phép khi đèn đỏ (`red_light_exceptions.allow_directions_on_red`)

**Ví dụ thật**: luật cho phép rẽ phải khi đèn đỏ (phổ biến ở nhiều nơi, với điều kiện nhường
đường). Với 1 lane CHUYÊN DỤNG chỉ rẽ phải:

```json
"rules": {
  "direction_mode": { "mode": "allow", "directions": ["right"] },
  "red_light_exceptions": { "allow_directions_on_red": ["right"] }
}
```

Xe cắt vạch dừng khi đèn đỏ ở lane này **sẽ không bị tính vi phạm**, vì hệ thống biết chắc xe
đang rẽ phải (lane chỉ có 1 hướng khả dĩ) và hướng đó nằm trong danh sách được phép khi đỏ.

### Ngoại lệ theo loại phương tiện (`red_light_exceptions.exempt_classes`)

**Ví dụ thật**: 1 số nơi cho phép xe máy đi tiếp khi đèn đỏ ở làn riêng dành cho xe máy rẽ
phải, trong khi ô tô vẫn phải dừng. Khai `"exempt_classes": ["motorcycle"]` — xe máy cắt vạch
lúc đèn đỏ sẽ không bị tính vi phạm, ô tô/xe tải vẫn bị tính bình thường.

### Giới hạn kỹ thuật cần biết

`allow_directions_on_red` và việc tra đèn theo hướng **chỉ hoạt động chắc chắn khi lane đó
CHỈ cho phép đúng 1 hướng** (`direction_mode.directions` là danh sách 1 phần tử). Lý do: tại
đúng thời điểm xe cắt vạch dừng, hệ thống chưa thể biết chắc xe sẽ rẽ hướng nào nếu lane cho
phép nhiều hướng cùng lúc (hướng thật chỉ xác định được sau, khi xe cắt qua vạch `end_direction`
— lúc đó đã quá muộn để quyết định có tính vi phạm đèn đỏ hay không). Nếu lane của bạn cho
nhiều hướng (vd vừa đi thẳng vừa rẽ phải chung 1 làn) thì các cơ chế theo-hướng này sẽ không
áp dụng được chính xác 100% — tạm thời chấp nhận giới hạn này, có thể cải thiện sau bằng cách
trì hoãn quyết định vi phạm đến khi biết hướng thật (chưa làm, cần cân nhắc thêm độ trễ cảnh
báo).

---

## Block `lines`

```json
{ "id": "SL1", "type": "stop_line", "lane_id": "L1", "points": [[x1,y1],[x2,y2]] }
{ "id": "ED1", "type": "end_direction", "direction": "left", "points": [[x1,y1],[x2,y2]] }
```

- `stop_line`: vạch dừng, PHẢI gán đúng `lane_id` — dùng cho rule vượt đèn đỏ.
- `end_direction`: vạch tại lối ra ngã tư, gán `direction` (`left`/`straight`/`right`) — dùng
  để xác định hướng đi thực tế của xe cho rule sai hướng rẽ.

## Block `vectors`

```json
{ "id": "V1", "type": "flow_direction", "start": [x1,y1], "end": [x2,y2], "lane_id": null }
```

`start` → `end` là hướng di chuyển ĐÚNG. Xe di chuyển ngược hướng này (góc > ~120°) trong
`wrong_way_zone` bị tính ngược chiều. `lane_id: null` (hoặc bỏ trống) = vector mặc định dùng
chung cho lane nào không có vector riêng; đặt `lane_id` cụ thể nếu ngã tư có nhiều hướng lưu
thông khác nhau cho từng lane (tránh so sai hướng giữa các lane khác nhau).

**Lưu ý khi tự vẽ**: chiều `start→end` phải đúng bằng mắt thường (xe đi từ điểm A đến điểm
B thật trong video) — đảo ngược 2 điểm này sẽ khiến MỌI xe đi đúng chiều bị báo ngược chiều
(gặp đúng lỗi này 2 lần trong lúc phát triển — dễ nhầm khi chỉnh tay).

## Block `traffic_lights`

```json
{
  "id": "TL_1",
  "mode": "detect",
  "traffic_light_zone_id": "TLZ_1",
  "apply_to": { "lane_ids": ["L1"], "directions": [] },
  "detect": {
    "hsv_ranges": { "red": [[0,120,70],[10,255,255],[170,120,70],[179,255,255]], "yellow": [[20,100,100],[30,255,255]], "green": [[40,50,50],[90,255,255]] },
    "min_pixels": 50
  },
  "manual": { "states": ["red","yellow","green"], "durations": {"red":6,"yellow":3,"green":5}, "start_offset": 0 }
}
```

- `mode: "detect"`: đọc màu thật qua pixel trong `traffic_light_zone_id` (cần vẽ zone type
  `traffic_light_zone` bao quanh đúng vị trí đèn). Dùng khi camera thấy rõ đèn.
- `mode: "manual"`: mô phỏng chu kỳ cố định theo `durations` (giây) — dùng khi đèn bị khuất/
  không nhìn rõ trong khung hình. Số pha tuỳ chỉnh được (2 pha đỏ-xanh hay 3 pha đều được,
  không ảnh hưởng `detect`).

## Block `system`

```json
"system": {
  "tracker_params": { "track_buffer": 60, "match_thresh": 0.95 },
  "detection_params": { "conf": 0.1, "imgsz": 1280, "nms_iou": 0.45, "min_visible_conf": 0.3 },
  "tracking_point": { "anchor": "bottom_center", "offset": [0, 0] },
  "features": {
    "detect_red_light": true,
    "detect_wrong_way": true,
    "detect_wrong_turn": true,
    "detect_wrong_lane": true
  },
  "wrong_way_config": { "zone_id": "WWZ_1", "min_move_distance": 30, "cos_threshold": -0.5 }
}
```

Mọi tham số tinh chỉnh tracker/detection nằm ở đây — **không có file YAML/config riêng nào
khác**, đúng 1 file JSON cho mỗi video. `features` cho phép tắt hẳn 1 rule nếu video/camera đó
không cần (tiết kiệm tài nguyên xử lý).

`tracking_point`: điểm đại diện cho vị trí xe dùng cho MỌI phép tính hình học (vào làn nào,
cắt vạch chưa...). Mặc định `bottom_center` (điểm giữa-đáy bbox, thường đúng với hầu hết góc
quay). Chỉ đổi sang `right_center` (hoặc khác) nếu đã kiểm chứng bằng mắt rằng điểm đó nằm
đúng vị trí xe thật trên MẶT ĐƯỜNG trong góc quay cụ thể của camera đó — **không copy giá trị
này từ config của video/camera khác mà không kiểm tra lại**, vì mỗi góc quay khác nhau, điểm
neo tính sai sẽ khiến toàn bộ rule dựa trên lane (đèn đỏ, sai làn, sai hướng) ngừng hoạt động
âm thầm (không báo lỗi, chỉ đơn giản là không bao giờ phát hiện được gì — đã gặp bug này khi
copy nguyên `right_center` từ hệ thống cũ sang mà chưa kiểm chứng).

### `display_params` — bật/tắt từng layer overlay vẽ lên video

```json
"system": {
  "display_params": {
    "show_zones": true, "show_lines": true, "show_trajectories": true, "show_vehicle_box": true,
    "show_plate_box": false, "show_helmet_box": false
  }
}
```

`show_zones`/`show_lines`/`show_trajectories`/`show_vehicle_box` không khai báo thì mặc định
`true` (giữ nguyên hành vi cũ). Chỉ ảnh hưởng PHẦN VẼ — không tắt được rule/logic tương ứng (vd tắt
`show_lines` vẫn tính đúng vượt đèn đỏ, chỉ là không vẽ vạch dừng lên video nữa). `show_vehicle_box`
tắt CẢ bbox lẫn ID/nhãn vi phạm của phương tiện (khối vẽ trong `draw_tracks`) — muốn video chỉ còn
đúng khung biển số/mũ bảo hiểm mà không bị bbox xe che khuất thì đặt cờ này `false`. Muốn video chỉ
còn bbox phương tiện (không zones/vạch/quỹ đạo) thì đặt 3 cờ đầu `false`, giữ `show_vehicle_box`.

`show_plate_box`/`show_helmet_box` mặc định `false` (khác 4 cờ trên) — vẽ thêm khung quanh đúng
vùng biển số đã đọc được (trắng) / vùng đầu-mũ bảo hiểm vừa kiểm tra (xanh = có mũ, đỏ = không
mũ). Chỉ có tác dụng khi `features.detect_plate`/`detect_helmet` tương ứng đang bật — model biển
số/mũ bảo hiểm chỉ chạy throttle (không phải mọi khung hình) nên khung được TÁI CHIẾU theo tỉ lệ
đã ghi nhận lần gần nhất lên bbox xe ở khung hình hiện tại, tự bám theo xe di chuyển giữa các lần
model chạy thật — không phải toạ độ cố định.

### `log_params.flow_bucket_seconds` — độ dài khung thời gian cho `traffic_flow.csv`

```json
"system": {
  "log_params": { "flow_bucket_seconds": 60 }
}
```

Chỉ áp dụng khi chạy có `log_name` (bật ghi log, xem `Pipeline.run()`). Quyết định mỗi dòng trong
`data/logs/<video>/traffic_flow.csv` đại diện cho bao nhiêu giây video (mặc định 60s/dòng — xem
`src/engine/logger.py::_write_traffic_flow`). Không khai báo thì dùng mặc định 60. Hạ xuống (vd
30s hoặc 10s) nếu muốn biểu đồ lưu lượng/hiệu năng theo thời gian mịn hơn (đổi lại nhiều dòng
hơn), tăng lên nếu video dài và chỉ cần xu hướng tổng quát.
