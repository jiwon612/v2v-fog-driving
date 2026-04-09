# ====== mqtt_receiver.py ======
import paho.mqtt.client as mqtt
import cv2
import numpy as np
import json
import base64
from config.settings import *

class MQTTReceiver:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        self.front_image = None
        self.front_image_timestamp = None
        self.front_speed = None
        self.front_speed_timestamp = None

    def on_connect(self, client, userdata, flags, rc):
        print(f"[MQTT Connected] result code {rc}")
        client.subscribe(FRONT_CAMERA_TOPIC)
        client.subscribe(SPEED_TOPIC)
        print(f"[MQTT Subscribed] {FRONT_CAMERA_TOPIC}, {SPEED_TOPIC}")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
        except json.JSONDecodeError:
            print(f"[ERROR] Invalid JSON on topic {msg.topic}")
            return

        if msg.topic == FRONT_CAMERA_TOPIC:
            try:
                # base64 → bytes → numpy → 이미지 복원
                decoded_bytes = base64.b64decode(payload["image"])
                nparr = np.frombuffer(decoded_bytes, np.uint8)
                self.front_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                self.front_image_timestamp = float(payload["timestamp"])
                print(f"[RECV] Image received at {self.front_image_timestamp:.2f}")
            except Exception as e:
                print(f"[ERROR] Failed to decode image: {e}")

        elif msg.topic == SPEED_TOPIC:
            try:
                self.front_speed = float(payload["speed"])
                self.front_speed_timestamp = float(payload["timestamp"])
                print(f"[RECV] Speed received: {self.front_speed:.2f} km/h at {self.front_image_timestamp:.2f}")
            except Exception as e:
                print(f"[ERROR] Failed to parse speed: {e}")

    def loop_forever(self):
        self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
        self.client.loop_forever()
