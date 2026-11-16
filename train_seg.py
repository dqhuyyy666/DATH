import os
import glob
from ultralytics import YOLO

def main():
    print("=" * 70)
    print("BẮT ĐẦU HUẤN LUYỆN YOLOv8-SEG (INSTANCE SEGMENTATION CHO Ô BẢNG GẤP NẾP)")
    print("=" * 70)

    # Dọn dẹp cache cũ
    for c_file in glob.glob("dataset_seg/labels/**/*.cache", recursive=True):
        try:
            os.remove(c_file)
        except Exception:
            pass

    dataset_yaml = "dataset_seg/dataset.yaml"
    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError("Chưa tìm thấy dataset_seg/dataset.yaml. Hãy chạy prepare_seg_dataset.py trước!")

    # Khởi tạo mô hình segmentation pre-trained
    model = YOLO("yolov8s-seg.pt")

    results = model.train(
        data=dataset_yaml,
        epochs=40,
        batch=4,
        imgsz=512,
        
        # Cấu hình chống học vẹt và bảo toàn hình học nếp gấp
        mosaic=0.0,
        mixup=0.0,
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        perspective=0.0,

        # Tăng cường chống nhiễu ánh sáng (giúp mô hình xử lý bóng đổ ở nếp gấp chữ V)
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,

        # Trọng số tối ưu
        box=7.5,
        cls=0.5,
        mask_ratio=4,
        
        lr0=0.005,
        lrf=0.01,
        warmup_epochs=2.0,
        cos_lr=True,
        patience=15,

        device="cpu",
        project="runs/seg",
        name="table_cell_seg_model",
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True
    )

    print("\n[✓] Huấn luyện hoàn tất! Trọng số lưu tại: runs/seg/table_cell_seg_model/weights/best.pt")

if __name__ == "__main__":
    main()