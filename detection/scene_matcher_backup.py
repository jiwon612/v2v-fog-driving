# ====== scene_matcher.py ======
import sys
sys.path.append('/home/guest/yolov5')

import time
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align# ====== scene_matcher.py ======
import sys
sys.path.append('/home/guest/yolov5')

import time
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
import numpy as np
import pandas as pd
import cv2
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.augmentations import letterbox
from utils.torch_utils import select_device
from concurrent.futures import ThreadPoolExecutor

class SceneMatcher:
    def __init__(self, model_path='best.pt', device='cuda:3', conf_thresh=0.8, sim_thresh=0.7):
        self.device = select_device(device)
        self.model = attempt_load(model_path, device=self.device)
        self.model.eval()
        self.model.conf = conf_thresh
        self.sim_thresh = sim_thresh
        # self.features와 hook 등록 제거: 병렬 충돌 방지 위해
        # self.features = []
        # self.model.model[4].register_forward_hook(self._hook_fn)

    # 제거: 이제 hook은 함수 내에서 직접 처리
    # def _hook_fn(self, module, input, output):
    #     self.features.append(output)

    def preprocess_image_array(self, img):
        img0 = img.copy()
        img, ratio, (pad_x, pad_y) = letterbox(img0, new_shape=(768, 576), auto=False, scaleup=False)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).float().to(self.device) / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        return img0, img_tensor, ratio, pad_x, pad_y

    def detect_objects(self, img_tensor):
        # 병렬 대응: hook을 지역변수로 등록하고, 끝나면 제거
        features_holder = []

        def hook_fn(module, input, output):
            features_holder.append(output)

        handle = self.model.model[4].register_forward_hook(hook_fn)  # hook 등록
        try:
            pred = self.model(img_tensor)[0]
            pred = non_max_suppression(pred, conf_thres=self.model.conf, iou_thres=0.45)[0]
            fmap = features_holder[0] if features_holder else None
        finally:
            handle.remove()  # hook 제거

        return pred, fmap

    def preds_to_boxes(self, preds):
        if preds is None or preds.size(0) == 0:
            return pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'cls'])
        data = preds[:, :6].cpu().numpy()
        return pd.DataFrame(data, columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'cls'])

    def extract_object_features(self, boxes, img_tensor, fmap):
        _, _, H, W = fmap.shape
        img_h, img_w = img_tensor.shape[2], img_tensor.shape[3]
        scale_x, scale_y = W / img_w, H / img_h
        
        rois = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.iloc[i][['xmin', 'ymin', 'xmax', 'ymax']]
            roi = torch.tensor([0, x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y], dtype=torch.float32)
            rois.append(roi)
        if not rois:
            return None
        rois = torch.stack(rois).to(self.device)
        pooled = roi_align(fmap, rois, output_size=(7, 7), spatial_scale=1.0)
        return pooled.view(pooled.size(0), -1)

    def match_scene(self, img1, img2):
        img0_a, tensor_a, ratio_a, pad_x_a, pad_y_a = self.preprocess_image_array(img1)
        img0_b, tensor_b, ratio_b, pad_x_b, pad_y_b = self.preprocess_image_array(img2)
        with torch.no_grad():
            preds_a, fmap_a = self.detect_objects(tensor_a)
            preds_b, fmap_b = self.detect_objects(tensor_b)
        boxes_a = self.preds_to_boxes(preds_a)
        boxes_b = self.preds_to_boxes(preds_b)
        if boxes_a.empty or boxes_b.empty:
            print("[MATCH] One of the images has no detected objects.")
            return 0.0
        emb_a = self.extract_object_features(boxes_a, tensor_a, fmap_a)
        emb_b = self.extract_object_features(boxes_b, tensor_b, fmap_b)
        if emb_a is None or emb_b is None:
            print("[MATCH] Failed to extract embeddings.")
            return 0.0
        sim_matrix = F.cosine_similarity(emb_a.unsqueeze(1), emb_b.unsqueeze(0), dim=2)
        matched = sum(sim_matrix[i].max().item() >= self.sim_thresh for i in range(sim_matrix.size(0)))
        match_ratio = matched / sim_matrix.size(0)
        print(f"[MATCH] Matched objects: {matched}/{sim_matrix.size(0)} ({match_ratio:.2%})")
        return match_ratio

    def draw_bbox(self, img):
        img0, img_tensor, ratio, pad_x, pad_y = self.preprocess_image_array(img)
        with torch.no_grad():
            preds, _ = self.detect_objects(img_tensor)

        if preds is None or preds.size(0) == 0:
            return img0

        for det in preds:
            x1, y1, x2, y2, conf, cls_id = det[:6].cpu().numpy()
            x1 = (x1 - pad_x) / ratio[0]
            y1 = (y1 - pad_y) / ratio[1]
            x2 = (x2 - pad_x) / ratio[0]
            y2 = (y2 - pad_y) / ratio[1]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            conf = float(conf)
            label = f"{self.model.names[int(cls_id)] if hasattr(self.model, 'names') else int(cls_id)} {conf:.2f}"
            cv2.rectangle(img0, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(img0, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img0

    def match_scene_multiple(self, front_memory_list, rear_img, max_workers=10):
        def match_one(mem):
            try:
                sim = self.match_scene(mem['front_image'], rear_img)
                return {'memory': mem, 'similarity': sim}
            except Exception as e:
                print(f"[ERROR] scene match 실패: {e}")
                return {'memory': None, 'similarity': -1.0}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(match_one, front_memory_list))

        valid_results = [r for r in results if r['memory'] is not None]
        if not valid_results:
            return None, 0.0

        best = max(valid_results, key=lambda r: r['similarity'])
        return best['memory'], best['similarity']

# 테스트 코드는 그대로
if __name__ == "__main__":
    matcher = SceneMatcher(model_path='/home/guest/yolov5/best.pt')
    img1 = cv2.imread('/home/guest/yolov5/sim_test/front.png')
    img2 = cv2.imread('/home/guest/yolov5/sim_test/back.png')

    start_time = time.time()
    ratio = matcher.match_scene(img1, img2)
    elapsed_time = time.time() - start_time
    print(f"[TIME] Scene match took {elapsed_time:.4f} seconds")

    if ratio >= 0.5:
        print("[RESULT] Same scene detected!")
    else:
        print("[RESULT] Different scenes.")

import numpy as np
import pandas as pd
import cv2
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.augmentations import letterbox
from utils.torch_utils import select_device
from concurrent.futures import ThreadPoolExecutor

class SceneMatcher:
    def __init__(self, model_path='best.pt', device='cuda:3', conf_thresh=0.8, sim_thresh=0.7):
        self.device = select_device(device)
        self.model = attempt_load(model_path, device=self.device)
        self.model.eval()
        self.model.conf = conf_thresh
        self.sim_thresh = sim_thresh
        self.features = []
        self.model.model[4].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.features.append(output)

    def preprocess_image_array(self, img):
        img0 = img.copy()
        img, ratio, (pad_x, pad_y) = letterbox(img0, new_shape=(768, 576), auto=False, scaleup=False)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).float().to(self.device) / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        return img0, img_tensor, ratio, pad_x, pad_y

    def detect_objects(self, img_tensor):
        self.features.clear()
        pred = self.model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres=self.model.conf, iou_thres=0.45)[0]
        fmap = self.features[0]
        return pred, fmap

    def preds_to_boxes(self, preds):
        if preds is None or preds.size(0) == 0:
            return pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'cls'])
        data = preds[:, :6].cpu().numpy()
        return pd.DataFrame(data, columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'cls'])

    def extract_object_features(self, boxes, img_tensor, fmap):
        _, _, H, W = fmap.shape
        img_h, img_w = img_tensor.shape[2], img_tensor.shape[3]
        scale_x, scale_y = W / img_w, H / img_h
        
        rois = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.iloc[i][['xmin', 'ymin', 'xmax', 'ymax']]
            roi = torch.tensor([0, x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y], dtype=torch.float32)
            rois.append(roi)
        if not rois:
            return None
        rois = torch.stack(rois).to(self.device)
        pooled = roi_align(fmap, rois, output_size=(7, 7), spatial_scale=1.0)
        return pooled.view(pooled.size(0), -1)

    def match_scene(self, img1, img2):
        img0_a, tensor_a, ratio_a, pad_x_a, pad_y_a = self.preprocess_image_array(img1)
        img0_b, tensor_b, ratio_b, pad_x_b, pad_y_b = self.preprocess_image_array(img2)
        with torch.no_grad():
            preds_a, fmap_a = self.detect_objects(tensor_a)
            preds_b, fmap_b = self.detect_objects(tensor_b)
        boxes_a = self.preds_to_boxes(preds_a)
        boxes_b = self.preds_to_boxes(preds_b)
        if boxes_a.empty or boxes_b.empty:
            print("[MATCH] One of the images has no detected objects.")
            return 0.0
        emb_a = self.extract_object_features(boxes_a, tensor_a, fmap_a)
        emb_b = self.extract_object_features(boxes_b, tensor_b, fmap_b)
        if emb_a is None or emb_b is None:
            print("[MATCH] Failed to extract embeddings.")
            return 0.0
        sim_matrix = F.cosine_similarity(emb_a.unsqueeze(1), emb_b.unsqueeze(0), dim=2)
        matched = sum(sim_matrix[i].max().item() >= self.sim_thresh for i in range(sim_matrix.size(0)))
        match_ratio = matched / sim_matrix.size(0)
        print(f"[MATCH] Matched objects: {matched}/{sim_matrix.size(0)} ({match_ratio:.2%})")
        return match_ratio

    def draw_bbox(self, img):
        img0, img_tensor, ratio, pad_x, pad_y = self.preprocess_image_array(img)  # img0: 원본 이미지 복사본
    
        with torch.no_grad():
            preds, _ = self.detect_objects(img_tensor)
    
        if preds is None or preds.size(0) == 0:
            return img0
    
        for det in preds:
            x1, y1, x2, y2, conf, cls_id = det[:6].cpu().numpy()
            # === De-letterbox ===
            x1 = (x1 - pad_x) / ratio[0]
            y1 = (y1 - pad_y) / ratio[1]
            x2 = (x2 - pad_x) / ratio[0]
            y2 = (y2 - pad_y) / ratio[1]
            
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            conf = float(conf)
            label = f"{self.model.names[int(cls_id)] if hasattr(self.model, 'names') else int(cls_id)} {conf:.2f}"
            cv2.rectangle(img0, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(img0, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
        return img0

    def match_scene_multiple(self, front_memory_list, rear_img, max_workers=6):
        """
        rear_img와 memory 리스트에 있는 각 front_image 간 유사도를 병렬로 비교하고,
        가장 유사한 이미지를 반환한다.
    
        Args:
            front_memory_list (List[Dict]): {'front_image': np.ndarray, ...} 형태의 리스트
            rear_img (np.ndarray): 현재 rear 차량의 프레임
            max_workers (int): 병렬 스레드 수
    
        Returns:
            best_match (dict), similarity (float)
        """
        def match_one(mem):
            try:
                sim = self.match_scene(mem['front_image'], rear_img)
                return {'memory': mem, 'similarity': sim}
            except Exception as e:
                print(f"[ERROR] scene match 실패: {e}")
                return {'memory': mem, 'similarity': 0.0}
    
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(match_one, front_memory_list))
    
        best = max(results, key=lambda r: r['similarity'])
        return best['memory'], best['similarity']
        
if __name__ == "__main__":
    matcher = SceneMatcher(model_path='/home/guest/yolov5/best.pt')
    
    # 예시 이미지 (실제 사용 시 numpy 배열 전달)
    img1 = cv2.imread('/home/guest/yolov5/sim_test/front.png')
    img2 = cv2.imread('/home/guest/yolov5/sim_test/back.png')
    
    start_time = time.time()
    
    ratio = matcher.match_scene(img1, img2)
    
    elapsed_time = time.time() - start_time
    print(f"[TIME] Scene match took {elapsed_time:.4f} seconds")
    
    if ratio >= 0.5:
        print("[RESULT] Same scene detected!")
    else:
        print("[RESULT] Different scenes.")