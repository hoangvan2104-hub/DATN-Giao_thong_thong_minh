# Chạy hệ thống bằng Docker (không cần cài Python/CUDA/torch thủ công)

Cách này đóng gói toàn bộ môi trường (Python, PyTorch bản CUDA đúng phiên bản, mọi thư viện) vào
1 Docker image — máy chạy chỉ cần cài Docker, không cần tự cài Python 3.10 + chọn tay bản torch
theo GPU như hướng dẫn cài trực tiếp ở [README.md mục 6](README.md#6-yêu-cầu-hệ-thống--cài-đặt).

## 0. Lưu ý quan trọng TRƯỚC khi build — model weights

`models/pretrained/*.pt` (model đã train/fine-tune) **không nằm trong git** (xem `.gitignore` —
cố ý loại bỏ, đúng thông lệ vì file weight quá nặng cho git thường). Nghĩa là:

- Nếu build ngay trên máy đã có sẵn `models/pretrained/yolo11s.pt`/`bestyolo.pt`/`LP_detector.pt`/
  `LP_ocr.pt` (máy đang phát triển đồ án) → build bình thường, `Dockerfile` copy thẳng các file
  này vào image, chạy ra kết quả đúng như chạy trực tiếp bằng Python.
- Nếu `git clone` sang máy MỚI rồi build ngay → `models/pretrained/` sẽ RỖNG, model sẽ lỗi
  "không tìm thấy file weight" ngay lúc detect. Phải copy tay 5 file model từ máy gốc sang
  `models/pretrained/` của máy mới TRƯỚC khi `docker build` (không có cơ chế tự tải nào khác).

Đây là giới hạn của cách phân phối qua git (dự án không dùng Git LFS), không phải lỗi của
Dockerfile — Docker chỉ đóng gói đúng những gì có sẵn trên đĩa lúc build.

## 1. Yêu cầu

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) hoặc Docker
  Engine + Docker Compose plugin (Linux).
- Muốn chạy có GPU (khuyến nghị — CPU vẫn chạy được nhưng không đạt thời gian thực): cần thêm
  driver NVIDIA mới + hỗ trợ GPU trong Docker:
  - **Windows**: bật WSL2 backend trong Docker Desktop (Settings → General), cài driver NVIDIA
    bản hỗ trợ WSL (driver thường của Windows là đủ, không cần cài driver riêng trong WSL) — xem
    [hướng dẫn chính thức NVIDIA](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
  - **Linux**: cài
    [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  - Không có GPU hoặc chưa cài được — xoá khối `deploy:` trong `docker-compose.yml`, hệ thống tự
    chuyển sang chạy CPU (chậm hơn, không lỗi — mọi model đều tự `torch.cuda.is_available()` rồi
    fallback, xem `tracker.py`/`no_helmet.py`/`plate_ocr.py`).

## 2. Build & chạy

```bash
docker compose up -d --build
```

Lần đầu build khá lâu (tải base image ~6-8GB đã kèm sẵn PyTorch+CUDA, cài thêm dependency) — các
lần sau chỉ build lại phần đổi (cache layer). Mở `http://localhost:8000`.

Xem log: `docker compose logs -f`. Dừng: `docker compose down` (dữ liệu trong `data/` và
`config/videos/` KHÔNG mất — 2 thư mục này mount thẳng từ máy host, xem giải thích trong
`docker-compose.yml`).

## 3. Thêm video mới khi chạy qua Docker

Giống hệt chạy trực tiếp — copy file video vào `data/input/` trên máy host (không cần vào trong
container), rồi vào Web UI dùng Config Wizard tạo cấu hình hoặc tải/kéo-thả/webcam ngay trên
"Màn hình giám sát chung". Vì `data/` được mount volume, Web UI ghi file y hệt vị trí chạy trực
tiếp bằng Python.

## 4. Giới hạn khi chạy qua Docker (so với chạy trực tiếp bằng Python)

- **Xem trực tiếp webcam**: Docker (đặc biệt trên Windows/WSL2) không có cách passthrough thiết bị
  USB webcam đơn giản/ổn định vào container — tính năng "Xem trực tiếp webcam" trên Web UI nhiều
  khả năng KHÔNG mở được thiết bị. Cần dùng webcam thì chạy trực tiếp bằng Python
  (`python run_web.py`) theo hướng dẫn ở README, không qua Docker.
- **Tốc độ lần đầu**: build image lần đầu cần internet để tải base image + cài dependency; sau khi
  build xong, hệ thống chạy được offline hoàn toàn (kể cả `ffmpeg` dùng để convert video kết quả
  sang H.264 — đã tải sẵn lúc build, xem comment trong `Dockerfile`).

## 5. Đổi cổng chạy

Sửa dòng `"8000:8000"` trong `docker-compose.yml` thành `"<cổng máy host>:8000"`, ví dụ `"9000:8000"`
để mở bằng `http://localhost:9000`.

## 6. Gửi cho người khác chạy — không cần tự build

`docker compose up -d --build` (mục 2) yêu cầu người nhận có TOÀN BỘ source code + 5 file model
weight (mục 0) rồi tự build — mất vài phút, có thể lỗi nếu thiếu file. Nếu muốn người khác **chỉ
double-click là chạy được, không cần build gì**, xuất thẳng image đã build sẵn trên máy bạn (đã có
đủ model + mọi thư viện bên trong):

```bash
docker save datn-traffic-web:latest -o datn-traffic-web.tar
```

File này nặng **~10GB+** — không gửi qua mạng được, phải chép qua USB/ổ cứng ngoài. Gửi kèm 1 thư
mục gồm:

```
datn-traffic-web.tar     (vừa xuất ở trên)
docker-compose.yml
run.bat                  (bấm đúp để chạy — tự nạp image + khởi động + mở trình duyệt)
stop.bat                 (bấm đúp để tắt)
data/input/               ← ít nhất 1 video demo (không thì mở lên chẳng có gì để chạy thử)
config/videos/            ← config tương ứng
```

`run.bat` tự kiểm tra: nếu chưa có GPU/driver NVIDIA thì tự chuyển sang chạy CPU thay vì báo lỗi
dừng hẳn. Người nhận chỉ cần: cài Docker Desktop (**vẫn là bước KHÔNG thể bỏ qua** — không có cách
đóng gói nào né được việc này, kể cả trên Windows Home còn cần thêm `wsl --install` + khởi động lại
máy 1 lần nếu WSL2 chưa có sẵn, xem mục 1) → mở Docker Desktop lên → double-click `run.bat`.

**Nếu nén cả thư mục trên thành 1 file `.zip` để gửi gọn hơn** (thường sẽ vượt 4GB do có
`datn-traffic-web.tar` bên trong): công cụ giải nén **có sẵn trong Windows ("Compressed Folders")
KHÔNG mở được** file `.zip` lớn dạng ZIP64 — báo lỗi "the compressed folder is invalid" dù file
hoàn toàn không hỏng (đã xác nhận bằng `7z t` — "Everything is Ok"). Người nhận cần cài
[7-Zip](https://www.7-zip.org/download.html) (miễn phí) rồi **chuột phải vào file `.zip` → 7-Zip →
Extract Here** (không double-click, sẽ gọi lại đúng công cụ Windows bị lỗi).

**Khi nào chọn cách này thay vì mục 2**: người nhận không rành kỹ thuật, không muốn tự gõ lệnh, cần
"cắm là chạy" (vd nộp bài kèm 1 ổ cứng gắn ngoài). Nếu người nhận là dân kỹ thuật có sẵn máy đang
dùng Docker thường xuyên thì cách mục 2 (qua GitHub) nhẹ hơn nhiều (~193MB so với ~10GB).
