# ====== mqtt_sender.py ======
import paho.mqtt.client as mqtt
import cv2
import json
import base64
from config.settings import *
from v2v_utils.helper import get_current_timestamp

class MQTTSender:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.connect(MQTT_BROKER, MQTT_PORT, 60)

    def send_image(self, image, topic, timestamp=None):
        # JPEG 인코딩
        _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        # base64 인코딩 → 문자열 변환
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        payload = {
            "timestamp": timestamp,
            "image": encoded_image
        }
        self.client.publish(topic, json.dumps(payload))

    def send_speed(self, speed):
        payload = {
            "timestamp": get_current_timestamp(),
            "speed": speed
        }
        self.client.publish(SPEED_TOPIC, json.dumps(payload))

    def send_lane(self, location, lane_id, topic):
        payload = {
            "timestamp": get_current_timestamp(),
            "location": {
                "x": location.x,
                "y": location.y,
                "z": location.z
            },
            "lane_id": lane_id
        }
        self.client.publish(topic, json.dumps(payload))

    def loop(self):
        self.client.loop_start()

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()
