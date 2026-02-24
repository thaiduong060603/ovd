# Sử dụng base image tối ưu cho Jetson Orin (L4T)
# Lưu ý: Chọn version phù hợp với JetPack trên máy bạn (ví dụ: r35.x.x)
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

# Thiết lập thư mục làm việc
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh
# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và hiển thị
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    pkg-config \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements và cài đặt (nếu có)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Copy toàn bộ code vào container
COPY . .

# Lệnh mặc định khi container khởi chạy
# (Chúng ta sẽ ghi đè lệnh này khi chạy docker run)
CMD ["python3", "main.py"]