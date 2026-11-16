import base64
import json
import time
import cv2
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.ocr.v20181119 import ocr_client, models

class TencentOCREngine:
    def __init__(self, secret_id: str, secret_key: str, region: str = "ap-guangzhou"):
        cred = credential.Credential(secret_id, secret_key)
        http_profile = HttpProfile()
        http_profile.endpoint = "ocr.tencentcloudapi.com"
        
        client_profile = ClientProfile()
        client_profile.httpProfile = http_profile
        self.client = ocr_client.OcrClient(cred, region, client_profile)

    def recognize_cell_crop(self, crop_img, data_type: str = "text") -> str:
        if crop_img is None or crop_img.size == 0:
            return ""

        success, encoded_img = cv2.imencode(".jpg", crop_img)
        if not success:
            return ""

        base64_str = base64.b64encode(encoded_img.tobytes()).decode("utf-8")
        req = models.GeneralBasicOCRRequest()
        req.from_json_string(json.dumps({"ImageBase64": base64_str}))

        try:
            resp = self.client.GeneralBasicOCR(req)
            resp_data = json.loads(resp.to_json_string())
            detections = resp_data.get("TextDetections", [])
            if not detections:
                return ""
            lines = [item["DetectedText"].strip() for item in detections if "DetectedText" in item]
            return " ".join(lines)
        except TencentCloudSDKException as err:
            if "RequestLimitExceeded" in str(err):
                time.sleep(0.1)
            return ""
        except Exception:
            return ""