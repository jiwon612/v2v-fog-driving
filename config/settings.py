# ====== settings.py ======

# ======================
# MQTT 설정
# ======================
MQTT_BROKER = 'localhost'
MQTT_PORT = 1883
FRONT_CAMERA_TOPIC = 'front/car/camera'
SPEED_TOPIC = 'front/car/speed'
LANE_TOPIC = "front2/lane"

# ======================
# YOLO 모델 설정
# ======================
YOLO_MODEL_PATH = 'model/best.pt'

# ======================
# 이미지 설정
# ======================
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
JPEG_QUALITY = 85

# ======================
# 카메라 설정
# ======================
CAMERA_SENSOR_TICK = 0.05
CAMERA_FOV = 90
CAMERA_LOCATION = {'x': 1.5, 'z': 2.4}
CAMERA_PITCH = -5.0

# ======================
# Publish 주기 설정
# ======================
PUBLISH_INTERVAL = 1.0

# ======================
# 기타 설정
# ======================
USE_GPU = True