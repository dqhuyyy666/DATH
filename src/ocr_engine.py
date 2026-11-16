import cv2
import easyocr

class TableOCREngine:
    def __init__(self, languages=None, gpu=False):
        if languages is None:
            languages = ["en"]
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def recognize_cell_crop(self, crop_img, data_type: str = "text") -> str:
        if crop_img is None or crop_img.size == 0:
            return ""

        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray, detail=0)
        if not results:
            return ""
        return " ".join(results).strip()