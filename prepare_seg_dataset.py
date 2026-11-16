import os
import glob
import json
import cv2
import numpy as np
from tqdm import tqdm

def build_seg_dataset(val_ratio=0.2):
    print("=" * 70)
    print("CHUẨN BỊ TẬP DỮ LIỆU SEGMENTATION (GIỮ NGUYÊN ẢNH RAW & POLYGON GẤP NẾP)")
    print("=" * 70)

    raw_dir = "data/raw"
    label_dir = "data/labels"
    base_dir = "dataset_seg"

    for split in ["train", "val"]:
        os.makedirs(os.path.join(base_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "labels", split), exist_ok=True)
        # Xóa cache cũ nếu có
        for c_file in glob.glob(os.path.join(base_dir, "labels", split, "*.cache")):
            os.remove(c_file)

    img_paths = sorted([p for p in glob.glob(os.path.join(raw_dir, "*.*")) 
                        if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

    valid_samples = []
    for p in img_paths:
        name = os.path.splitext(os.path.basename(p))[0]
        json_p = os.path.join(label_dir, f"{name}.json")
        if os.path.exists(json_p):
            valid_samples.append((p, json_p, name))

    print(f"[*] Tìm thấy {len(valid_samples)} mẫu hợp lệ (có đủ ảnh raw và file JSON).")
    if len(valid_samples) == 0:
        print("[!] Không tìm thấy dữ liệu. Hãy kiểm tra lại thư mục data/raw và data/labels.")
        return

    np.random.seed(42)
    indices = np.random.permutation(len(valid_samples))
    val_count = max(1, int(len(valid_samples) * val_ratio))
    val_indices = set(indices[:val_count])

    processed_count = 0
    total_polygons = 0

    for idx, (img_p, json_p, name) in enumerate(tqdm(valid_samples, desc="Chuyển đổi nhãn Polygon")):
        img = cv2.imread(img_p)
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(json_p, "r", encoding="utf-8") as f:
            ann_data = json.load(f)

        split = "val" if idx in val_indices else "train"
        
        # Sao chép nguyên bản ảnh raw (không làm biến dạng, giữ nguyên nếp gấp)
        cv2.imwrite(os.path.join(base_dir, "images", split, f"{name}.jpg"), img)

        cells = ann_data.get("cells", ann_data.get("annotations", ann_data.get("shapes", [])))
        seg_lines = []

        for cell in cells:
            pts = cell.get("points", [])
            if len(pts) < 3:
                continue

            # Chuẩn hóa tọa độ các đỉnh đa giác về dải [0, 1]
            norm_coords = []
            for pt in pts:
                px, py = float(pt[0]), float(pt[1])
                nx = np.clip(px / w, 0.0, 1.0)
                ny = np.clip(py / h, 0.0, 1.0)
                norm_coords.extend([f"{nx:.6f}", f"{ny:.6f}"])

            seg_lines.append("0 " + " ".join(norm_coords))
            total_polygons += 1

        with open(os.path.join(base_dir, "labels", split, f"{name}.txt"), "w", encoding="utf-8") as f_lbl:
            f_lbl.write("\n".join(seg_lines))

        processed_count += 1

    # Tạo file dataset.yaml
    yaml_path = os.path.join(base_dir, "dataset.yaml")
    abs_base = os.path.abspath(base_dir).replace("\\", "/")
    with open(yaml_path, "w", encoding="utf-8") as f_yaml:
        f_yaml.write(f"path: {abs_base}\ntrain: images/train\nval: images/val\n\nnames:\n  0: cell\n")

    print(f"\n[✓] Hoàn tất chuẩn bị dataset Segmentation!")
    print(f"    - Tổng ảnh xử lý : {processed_count} (Train: {processed_count - val_count}, Val: {val_count})")
    print(f"    - Tổng số polygon: {total_polygons}")
    print(f"    - Cấu hình lưu tại: {yaml_path}")

if __name__ == "__main__":
    build_seg_dataset()