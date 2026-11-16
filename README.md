# 📄 Hệ thống Trích xuất Dữ liệu Bảng biểu từ Tài liệu Gập nếp (V-Fold Table Extraction)

Dự án xây dựng quy trình tự động phát hiện, phân đoạn cấu trúc ô bảng và trích xuất văn bản thành định dạng Excel (`.xlsx`) từ hình ảnh tài liệu thực tế bị biến dạng do nếp gấp chữ V (V-folds). Hệ thống kết hợp mô hình **YOLOv8-Seg** (Instance Segmentation), kỹ thuật cắt gọt mặt nạ cục bộ (Mask-based Cropping), thuật toán gom nhóm hình học và **Tencent Cloud OCR**.

---

## 1. Kiến trúc hệ thống (Pipeline Architecture)

Quy trình xử lý dữ liệu đi qua 4 giai đoạn chính:

1. **Chuẩn hóa dữ liệu (`prepare_seg_dataset.py`):** Chuyển đổi nhãn đa giác (Polygon) từ file JSON gốc sang định dạng chuẩn của YOLOv8-Seg (tọa độ chuẩn hóa $[0, 1]$), chia tập dữ liệu theo tỷ lệ Train/Val.
2. **Huấn luyện mô hình (`train_seg.py`):** Huấn luyện mạng `yolov8s-seg` để học biên dạng đa giác uốn lượn theo nếp gấp giấy. Tắt các phép tăng cường làm biến dạng hình học (`mosaic`, `mixup`, `degrees`, `perspective`) để bảo toàn đặc trưng V-fold.
3. **Suy luận & Cắt gọt mặt nạ (`infer_seg.py`):**
   * Dự đoán mặt nạ đa giác của từng ô bảng trên ảnh thực tế.
   * Sử dụng `cv2.fillPoly` để cô lập nội dung chữ bên trong từng đa giác trên nền trắng, loại bỏ bóng đổ và chữ nhiễu từ các ô lân cận.
4. **Hậu xử lý hình học & Xuất Excel (`ocr_engine.py`, `exporter.py`):** Gom nhóm các ô thành hàng và cột dựa trên độ gối chiều cao (Vertical Overlap) và tọa độ trục hoành, gửi ảnh ô qua API OCR để nhận diện văn bản và xuất ra bảng tính `.xlsx`.

---

## 2. Cấu trúc thư mục (Project Structure)

    DATH/
    ├── configs/
    │   └── dataset.yaml           # Cấu hình đường dẫn dữ liệu Train/Val và danh sách lớp
    ├── datasets/                  # Thư mục chứa dữ liệu ảnh và nhãn sau khi chuẩn hóa
    ├── src/
    │   ├── exporter.py            # Xuất dữ liệu đã tái cấu trúc ra file Excel (.xlsx)
    │   ├── ocr_engine.py          # Xử lý logic nhận diện văn bản và gom nhóm hàng/cột
    │   ├── tencent_ocr.py         # Module kết nối API Tencent Cloud OCR
    │   └── utils.py               # Các hàm tiện ích xử lý ảnh và hình học
    ├── infer_seg.py               # Pipeline suy luận, cắt mask, gọi OCR và xuất Excel
    ├── prepare_seg_dataset.py     # Script chuyển đổi nhãn JSON sang YOLO Segmentation
    ├── train_seg.py               # Script cấu hình và huấn luyện mô hình YOLOv8-Seg
    ├── runs/                      # Thư mục lưu trữ trọng số (best.pt) và biểu đồ huấn luyện
    ├── requirements.txt           # Danh sách các thư viện Python cần thiết
    └── README.md                  # Tài liệu hướng dẫn sử dụng dự án

---

## 3. Cài đặt môi trường (Installation)

**Yêu cầu hệ thống:** Python 3.8 trở lên (Khuyến nghị sử dụng môi trường có hỗ trợ GPU NVIDIA CUDA hoặc chạy trên Google Colab).

**Bước 1: Clone dự án về máy**

    git clone https://github.com/dqhuyyy666/DATH.git
    cd DATH

**Bước 2: Cài đặt các thư viện phụ thuộc**

    pip install ultralytics opencv-python numpy pandas openpyxl tencentcloud-sdk-python

---

## 4. Hướng dẫn sử dụng (Usage)

### Bước 1: Chuẩn bị dữ liệu
Đặt ảnh gốc và file nhãn `.json` vào thư mục dữ liệu thô, sau đó chạy script chuẩn hóa để tạo cấu trúc dataset cho YOLO:

    python prepare_seg_dataset.py

### Bước 2: Huấn luyện mô hình
Kiểm tra đường dẫn trong `configs/dataset.yaml` và chạy lệnh huấn luyện:

    python train_seg.py

> **Lưu ý:** Nếu chạy trên máy có card đồ họa NVIDIA hoặc Google Colab, đảm bảo tham số `device="0"`. Sau khi hoàn tất, tệp trọng số tốt nhất sẽ được lưu tại `runs/seg/table_cell_seg_fast/weights/best.pt`.

### Bước 3: Chạy suy luận và xuất file Excel
Cấu hình API Key của Tencent Cloud OCR và đường dẫn tới file trọng số `best.pt` trong file `src/infer_seg.py`, sau đó thực thi:

    python infer_seg.py

Kết quả đầu ra bao gồm ảnh trực quan hóa đa giác (debug) và file bảng tính `.xlsx` được tái cấu trúc đầy đủ hàng và cột.
