FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép và cài đặt thư viện phụ thuộc trước để tối ưu cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn và dữ liệu vào container
COPY . .

# Cloud Run sử dụng biến môi trường PORT (thường là 8080)
ENV PORT=8080

# Expose cổng
EXPOSE 8080

# Chạy ứng dụng FastAPI
CMD ["python", "main.py"]
