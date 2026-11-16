import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from src.utils import load_yaml_config
from src.tencent_ocr import TencentOCREngine
from src.ocr_engine import TableOCREngine
from src.exporter import export_cells_to_excel


def reassign_rows_cols_by_geometry(parsed_cells):
    """Gom hàng theo độ gối chiều cao và sắp xếp cột theo trục X."""
    if not parsed_cells:
        return []

    cells_sorted = sorted(parsed_cells, key=lambda c: np.min(c["points"][:, 1]))
    
    rows = []
    for cell in cells_sorted:
        y1 = float(np.min(cell["points"][:, 1]))
        y2 = float(np.max(cell["points"][:, 1]))
        h = max(1.0, y2 - y1)
        
        placed = False
        for row in rows:
            r_y1 = np.mean([np.min(c["points"][:, 1]) for c in row])
            r_y2 = np.mean([np.max(c["points"][:, 1]) for c in row])
            
            overlap = max(0.0, min(y2, r_y2) - max(y1, r_y1))
            if (overlap / h) > 0.40:
                row.append(cell)
                placed = True
                break
        if not placed:
            rows.append([cell])

    reassigned = []
    for r_idx, row in enumerate(rows, start=1):
        row_sorted_x = sorted(row, key=lambda c: np.min(c["points"][:, 0]))
        for c_idx, cell in enumerate(row_sorted_x, start=1):
            cell["row"] = r_idx
            cell["col"] = c_idx
            reassigned.append(cell)
            
    return reassigned


def crop_polygon_cell(raw_img, points):
    # 1. Chuyển points về dạng numpy array
    pts = np.array(points, dtype=np.int32)
    
    # 2. Lấy bounding box bao quanh đa giác
    x, y, w, h = cv2.boundingRect(pts)
    
    # Giới hạn bounding box không vượt quá kích thước ảnh gốc
    img_h, img_w = raw_img.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    
    # Cắt ảnh gốc dựa theo tọa độ an toàn
    cropped_img = raw_img[y1:y2, x1:x2]
    crop_h, crop_w = cropped_img.shape[:2]
    
    # 3. Tạo mask có kích thước CHÍNH XÁC bằng với cropped_img vừa cắt (loại bỏ hoàn toàn lệch 1 pixel)
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    
    # Dịch chuyển tọa độ polygon về hệ tọa độ cục bộ của ô crop
    shifted_pts = pts - np.array([x1, y1])
    cv2.fillPoly(mask, [shifted_pts], 255)
    
    # 4. Tạo nền trắng cùng kích thước với cropped_img
    white_bg = np.ones_like(cropped_img) * 255
    
    # 5. Kết hợp mask và ảnh (lúc này các shape đã đồng bộ 100%: (crop_h, crop_w))
    clean_cell = np.where(mask[:, :, None] == 255, cropped_img, white_bg)
    
    return clean_cell

def run_seg_pipeline():
    print("=" * 70)
    print("SUY LUẬN YOLOv8-SEG: RAW IMAGE -> POLYGON MASK -> OCR -> EXCEL")
    print("=" * 70)

    cfg = load_yaml_config("configs/pipeline_config.yaml")
    raw_dir = cfg.get("paths", {}).get("raw_dir", "data/raw")
    output_dir = "data/outputs_seg"
    debug_preds_dir = "data/debug_seg_preds"

    for d in [output_dir, debug_preds_dir]:
        os.makedirs(d, exist_ok=True)

    weight_path = "runs/seg/table_cell_seg_model/weights/best.pt"
    if not os.path.exists(weight_path):
        print(f"[!] Không tìm thấy trọng số tại {weight_path}. Hãy chạy train_seg.py trước!")
        return

    print(f"[*] Đang nạp mô hình Segmentation: {weight_path}")
    model = YOLO(weight_path)

    # Khởi tạo OCR engine
    ocr_type = cfg.get("ocr", {}).get("engine", "tencent")
    if ocr_type == "tencent":
        t_cfg = cfg.get("ocr", {}).get("tencent", {})
        ocr_engine = TencentOCREngine(
            secret_id=t_cfg.get("secret_id", ""),
            secret_key=t_cfg.get("secret_key", ""),
            region=t_cfg.get("region", "ap-guangzhou")
        )
    else:
        ocr_engine = TableOCREngine(languages=["en"], gpu=False)

    img_paths = sorted([p for p in glob.glob(os.path.join(raw_dir, "*.*")) 
                        if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

    success_count = 0
    for img_p in tqdm(img_paths, desc="Xử lý ảnh"):
        base_name = os.path.basename(img_p)
        name_no_ext = os.path.splitext(base_name)[0]
        excel_out = os.path.join(output_dir, f"{name_no_ext}.xlsx")

        raw_img = cv2.imread(img_p)
        if raw_img is None:
            continue

        # Dự đoán Segmentation trên ảnh raw
        results = model.predict(raw_img, conf=0.25, iou=0.45, verbose=False)[0]

        # Lưu ảnh vẽ polygon kiểm tra
        debug_plot = results.plot()
        cv2.imwrite(os.path.join(debug_preds_dir, f"seg_{base_name}"), debug_plot)

        if results.masks is None or len(results.masks) == 0:
            continue

        polygons = results.masks.xy  # Danh sách tọa độ các đỉnh polygon cho từng ô
        parsed_cells = []

        for idx, poly in enumerate(polygons):
            if len(poly) < 3:
                continue
            parsed_cells.append({
                "id": idx,
                "row": 1,
                "col": 1,
                "points": np.array(poly, dtype=np.float32)
            })

        if not parsed_cells:
            continue

        # Gom hàng và sắp xếp cột theo hình học
        parsed_cells = reassign_rows_cols_by_geometry(parsed_cells)
        parsed_cells = sorted(parsed_cells, key=lambda c: (c["row"], c["col"]))

        # Cắt ô theo mặt nạ polygon và gọi OCR
        for cell in parsed_cells:
            crop = crop_polygon_cell(raw_img, cell["points"])
            cell["text"] = ocr_engine.recognize_cell_crop(crop)

        # Xuất Excel
        export_cells_to_excel(parsed_cells, excel_out)
        success_count += 1

    print("\n" + "=" * 70)
    print(f"[✓] ĐÃ XỬ LÝ XONG {success_count}/{len(img_paths)} ẢNH.")
    print(f"Ảnh vẽ Mask Polygon : {debug_preds_dir}")
    print(f"File Excel kết quả  : {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    run_seg_pipeline()