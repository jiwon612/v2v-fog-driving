# ====== scene_matcher.py ======
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
            return 0.0, 1.0
    
        # (1) 사용할 class 이름
        base_classes = ["tree", "streetlamp"]
        event_classes = ["streetsign01", "streetsign04", "advertisement", "waringaccident"]
                
        # (2) 이름→index 변환
        name_to_idx = {v: int(k) for k, v in self.model.names.items()}
        base_class_idx = set([name_to_idx[n] for n in base_classes if n in name_to_idx])
        event_class_idx = set([name_to_idx[n] for n in event_classes if n in name_to_idx])
    
        # (3) 공통 event class 추출 (strict하지 않게)
        set_a_event = set(boxes_a['cls'].astype(int)) & event_class_idx
        set_b_event = set(boxes_b['cls'].astype(int)) & event_class_idx
        common_event_classes = set_a_event & set_b_event
        
        if not common_event_classes:
            print("[MATCH] No base or common event classes to compare.")
            return 0.0, 1.0    
            
        match_classes = set(base_class_idx) | common_event_classes

        total_match = 0
        total_cnt = 0
        bbox_ratios = []
        
        for cls_idx in match_classes:
            sub_a = boxes_a[boxes_a['cls'] == cls_idx]
            sub_b = boxes_b[boxes_b['cls'] == cls_idx]
            if sub_a.empty or sub_b.empty:
                continue
            emb_a = self.extract_object_features(sub_a, tensor_a, fmap_a)
            emb_b = self.extract_object_features(sub_b, tensor_b, fmap_b)
            if emb_a is None or emb_b is None:
                continue
            sim_matrix = F.cosine_similarity(emb_a.unsqueeze(1), emb_b.unsqueeze(0), dim=2)
            matched = sum(sim_matrix[i].max().item() >= self.sim_thresh for i in range(sim_matrix.size(0)))
            total_match += matched
            total_cnt += sim_matrix.size(0)
            class_name = self.model.names[cls_idx] if hasattr(self.model, 'names') else str(cls_idx)
            print(f"[MATCH][{class_name}] Matched: {matched}/{sim_matrix.size(0)}")

            for i in range(sim_matrix.size(0)):
                max_sim, max_j = sim_matrix[i].max(0)
                if max_sim.item() >= self.sim_thresh:

                    # bbox area for matched pair
                    x1a, y1a, x2a, y2a = sub_a.iloc[i][['xmin', 'ymin', 'xmax', 'ymax']]
                    x1b, y1b, x2b, y2b = sub_b.iloc[max_j.item()][['xmin', 'ymin', 'xmax', 'ymax']]
                    area_a = (x2a - x1a) * (y2a - y1a)
                    area_b = (x2b - x1b) * (y2b - y1b)

                    if area_b > 0:
                        bbox_ratios.append(area_b / area_a)
    
        if total_cnt == 0 or len(bbox_ratios) == 0:
            return 0.0, 1.0
        match_ratio = total_match / total_cnt
        avg_bbox_ratio = sum(bbox_ratios) / len(bbox_ratios)
        print(f"[MATCH] Total matched target objects: {total_match}/{total_cnt} ({match_ratio:.2%})")
        return match_ratio, avg_bbox_ratio 

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

    def match_scene_multiple(self, front_memory_list, rear_img, max_workers=20):
        def match_one(mem):
            try:
                sim, ratio = self.match_scene(mem['front_image'], rear_img)
                return {'memory': mem, 'similarity': sim, 'ratio': ratio}
            except Exception as e:
                print(f"[ERROR] scene match 실패: {e}")
                return {'memory': None, 'similarity': -1.0, 'ratio': 1.0}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(match_one, front_memory_list))

        valid_results = [r for r in results if r['memory'] is not None]
        if not valid_results:
            return None, 0.0, 1.0

        best = max(valid_results, key=lambda r: r['similarity'])
        return best['memory'], best['similarity'], best['ratio']

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
