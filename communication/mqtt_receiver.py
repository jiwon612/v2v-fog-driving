# ====== mqtt_receiver.py ======
import paho.mqtt.client as mqtt
import cv2
import numpy as np
import json
import base64
from config.settings import *
import carla

class MQTTReceiver:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        self.front_image = None
        self.front_image_timestamp = None
        self.front_speed = None
        self.front_speed_timestamp = None
        self.front_lane_id = None
        self.front_lane_type = None
        self.front_lane_timestamp = None

        # front2 정보
        self.front2_lane_id = None
        self.front2_lane_timestamp = None
        self.front2_location = None

    def on_connect(self, client, userdata, flags, rc):
        print(f"[MQTT Connected] result code {rc}")
        client.subscribe(FRONT_CAMERA_TOPIC)
        client.subscribe(SPEED_TOPIC)
        client.subscribe(LANE_TOPIC)

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
            except Exception as e:
                print(f"[ERROR] Failed to decode image: {e}")

        elif msg.topic == SPEED_TOPIC:
            try:
                self.front_speed = float(payload["speed"])
                self.front_speed_timestamp = float(payload["timestamp"])
            except Exception as e:
                print(f"[ERROR] Failed to parse speed: {e}")
        
        elif msg.topic == LANE_TOPIC:
            try:
                self.front2_lane_timestamp = float(payload["timestamp"])
                loc = payload.get("location", {})
                self.front2_location = carla.Location(
                    x=float(loc.get("x", 0.0)),
                    y=float(loc.get("y", 0.0)),
                    z=float(loc.get("z", 0.0))
                )
                self.front2_lane_id = int(payload["lane_id"])
            except Exception as e:
                print(f"[ERROR] Failed to parse front2 lane: {e}")


    def loop_forever(self):
        self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
        self.client.loop_forever()
