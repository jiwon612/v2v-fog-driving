import sys
sys.path.append('/home/guest/yolov5')

import torch
import cv2
import os
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.augmentations import letterbox
from utils.torch_utils import select_device
import numpy as np

class CarDetector:
    def __init__(self, model_path='./yolov5s.pt', device='cuda:0', conf_thresh=0.8):
        self.device = select_device(device)
        self.model = attempt_load(model_path, device=self.device)
        self.model.eval()
        self.model.conf = conf_thresh

        self.target_classes = ['car', 'truck', 'bus']
        self.target_ids = [i for i, name in self.model.names.items() if name in self.target_classes]

    def preprocess(self, img):
        img0 = img.copy()
        img, ratio, (pad_x, pad_y) = letterbox(img0, new_shape=(640, 640), auto=False, scaleup=False)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).float().to(self.device) / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        return img0, img_tensor, ratio, pad_x, pad_y

    def detect_and_draw(self, img, save_path=None):
        img0, img_tensor, ratio, pad_x, pad_y = self.preprocess(img)

        with torch.no_grad():
            pred = self.model(img_tensor)[0]
            pred = non_max_suppression(pred, conf_thres=self.model.conf, iou_thres=0.45)[0]

        if pred is None or pred.size(0) == 0:
            return False, img0

        detected = False
        for det in pred:
            x1, y1, x2, y2, conf, cls_id = det[:6].cpu().numpy()
            if int(cls_id) not in self.target_ids:
                continue

            # 좌표 복원
            x1 = (x1 - pad_x) / ratio[0]
            y1 = (y1 - pad_y) / ratio[1]
            x2 = (x2 - pad_x) / ratio[0]
            y2 = (y2 - pad_y) / ratio[1]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # bbox + label 그리기
            label = f"{self.model.names[int(cls_id)]} {conf:.2f}"
            cv2.rectangle(img0, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(img0, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            detected = True

        if save_path and detected:
            cv2.imwrite(save_path, img0)

        return detected, img0
