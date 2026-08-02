# Đóng gói hệ thống giám sát vi phạm giao thông (Web UI) thành 1 image Docker chạy được ngay,
# không cần cài Python/CUDA/torch thủ công trên máy đích. Xem DOCKER.md để biết cách build/chạy
# và các lưu ý quan trọng (đặc biệt: model weights KHÔNG nằm trong git, phải build từ máy đã có
# sẵn models/pretrained/ — xem DOCKER.md mục "Model weights").
#
# Base image chọn ĐÚNG bản torch==2.6.0+cu124 đã kiểm chứng chạy thật xuyên suốt đồ án (xem
# README.md mục 6) — tránh phải tự cài CUDA toolkit + chọn tay bản torch tương thích như khi cài
# trên máy trần, giảm hẳn rủi ro lệch phiên bản.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# libGL/libglib cho opencv-python (bản đầy đủ, không phải headless — giữ nguyên để không ảnh
# hưởng nhánh code cv2.imshow của --show nếu sau này chạy có X11 forwarding); libsm/libxext/
# libxrender cho vài thư viện xử lý ảnh phụ thuộc X11 headers dù không thực sự mở cửa sổ.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Cài dependency trước, tách riêng layer khỏi COPY code — code đổi liên tục nhưng
# requirements.txt hiếm khi đổi, cache layer pip install giữa các lần build lại.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Tải sẵn ffmpeg tĩnh của imageio-ffmpeg NGAY LÚC BUILD (không phải lúc chạy) — state.py::
# _reencode_for_browser() cần binary này để convert video kết quả sang H.264 phát được trên
# trình duyệt; nếu để tới lúc chạy mới tải, container không có internet (demo air-gapped) sẽ
# lỗi ngay lần xử lý video đầu tiên.
RUN python -c "import imageio_ffmpeg; imageio_ffmpeg.get_ffmpeg_exe()"

# Copy toàn bộ source — .dockerignore đã loại data/ (2GB+ dữ liệu test, mount qua volume thay
# vì bake vào image), docs/eval/train/báo cáo (không cần lúc CHẠY), model benchmark-only nặng
# (models/pretrained/helmet/, *.onnx chưa dùng).
COPY . .

# Tạo sẵn cây thư mục data/ rỗng (đề phòng chưa mount volume lần đầu chạy thử) — khớp đúng quy
# ước thư mục của dự án (xem CLAUDE.md mục "Cấu trúc thư mục & quy ước chạy").
RUN mkdir -p data/input data/output data/logs data/evidence data/thumbnails data/processed

EXPOSE 8000

CMD ["python", "run_web.py"]
