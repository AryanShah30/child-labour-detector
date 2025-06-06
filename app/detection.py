from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

model = YOLO("yolov8n.pt")

def run_yolo_inference(pil_img):
    image = np.array(pil_img.convert("RGB"))
    results = model(image)
    result_img = results[0].plot()
    crops = extract_person_crops(results[0], image)
    return result_img, results[0], crops

def extract_person_crops(result, orig_image):
    crops = []
    boxes = result.boxes
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        if cls_id == 0:  
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            crop = orig_image[y1:y2, x1:x2]
            crops.append(crop)
    return crops