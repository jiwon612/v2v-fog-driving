# -*- coding: utf-8 -*-

import sys
import time
import carla
import random
import numpy as np
import math
import os
import cv2
from datetime import datetime
import csv
import os

from communication.mqtt_sender import MQTTSender
from communication.mqtt_receiver import MQTTReceiver
from sensors.camera_manager import CameraManager
from detection.scene_matcher import SceneMatcher
from control.controller import Controller
from v2v_utils.helper import get_current_timestamp, calculate_distance, pretty_print_distance
from config.settings import *
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append('/home/guest/carla-simulator/PythonAPI/carla')
sys.path.append('/home/guest/carla-simulator/PythonAPI')
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.local_planner import RoadOption

def spawn_vehicle(world, blueprint, a, b, c):
    map = world.get_map()
    origin_loc = carla.Location(x=a, y=b, z=c)
    base_wp = map.get_waypoint(origin_loc, project_to_road=True, lane_type=carla.LaneType.Driving)

    transform = base_wp.transform
    transform.location.z += 0.3  # 바닥 관통 방지

    vehicle = world.try_spawn_actor(blueprint, transform)
    if vehicle:
        print(f"[SPAWN SUCCESS] 차량 스폰 완료 (lane_id={base_wp.lane_id})")
        return vehicle
    else:
        raise RuntimeError("[SPAWN FAIL] 차량 스폰 실패")
        
def spawn_props(world, blueprint_library, csv_path="./prop_spawn_points_20m.csv"):
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV 파일이 존재하지 않습니다: {csv_path}")
        return

    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            x = float(row["x"])
            y = float(row["y"])
            z = float(row["z"])
            prop_id = row["prop_id"]

            blueprint = blueprint_library.find(prop_id)
            if blueprint is None:
                print(f"[SKIPPED] {prop_id} blueprint 찾을 수 없음")
                continue

            location = carla.Location(x=x, y=y, z=z)
            transform = carla.Transform(location, carla.Rotation(yaw=0.0))
            prop = world.try_spawn_actor(blueprint, transform)

            if prop:
                print(f"[SPAWNED] {prop_id} at x={x:.2f}, y={y:.2f}, z={z:.2f}")
            else:
                print(f"[FAILED] {prop_id} at x={x:.2f}, y={y:.2f}, z={z:.2f}")

def main():
    actor_list = []
    
    # 현재 날짜와 시간 정보 가져오기
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")

    # 폴더명 생성
    save_dir = f"./scene_match/{timestamp_str}"
    os.makedirs(save_dir, exist_ok=True)

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        world = client.load_world('Town04')
        blueprint_library = world.get_blueprint_library()

        weather = world.get_weather()
        weather.fog_density = 60.0
        weather.fog_distance = 20.0
        weather.fog_falloff = 1.0
        weather.precipitation = 0.0
        weather.wetness = 0.0
        weather.sun_altitude_angle = 30.0
        world.set_weather(weather)
        print("[WEATHER] 짙은 안개만 설정 완료 (비 없음, 밝은 배경)")

        spawn_props(world, blueprint_library)

        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        rear_vehicle = spawn_vehicle(world, vehicle_bp, 205.74, 11.27, 0.3)
        front_vehicle = spawn_vehicle(world, vehicle_bp, 145.74, 10.46, 0.3)

        actor_list.extend([front_vehicle, rear_vehicle])

        front_agent = BehaviorAgent(front_vehicle, behavior='normal')
        rear_agent = BehaviorAgent(rear_vehicle, behavior='normal')

        map = world.get_map()
        start_wp = map.get_waypoint(front_vehicle.get_location())
        route = []
        current_wp = start_wp
        for _ in range(100):
            route.append(current_wp)
            next_wps = current_wp.next(5.0)
            if not next_wps:
                break
            current_wp = next_wps[0]

        front_agent.set_global_plan([(wp, RoadOption.LANEFOLLOW) for wp in route])

        rear_start_wp = map.get_waypoint(rear_vehicle.get_location())
        rear_route = []
        current_wp = rear_start_wp
        for _ in range(100):
            rear_route.append(current_wp)
            next_wps = current_wp.next(5.0)
            if not next_wps:
                break
            current_wp = next_wps[0]
        rear_agent.set_global_plan([(wp, RoadOption.LANEFOLLOW) for wp in rear_route])

        front_agent.set_target_speed(70)
        rear_agent.set_target_speed(70)

        front_front_camera = CameraManager(world, front_vehicle, carla.Transform(carla.Location(x=1.5, z=2.4)))
        rear_front_camera = CameraManager(world, rear_vehicle, carla.Transform(carla.Location(x=1.5, z=2.4)))

        front_sender = MQTTSender()
        front_sender.loop()

        rear_receiver = MQTTReceiver()
        rear_receiver.client.connect(MQTT_BROKER, MQTT_PORT, 60)
        rear_receiver.client.loop_start()
        
        scene_matcher = SceneMatcher(model_path=YOLO_MODEL_PATH, device='cuda:3')
        controller = Controller()
        executor = ThreadPoolExecutor(max_workers=4)  # 병렬 비교 스레드 풀

        rear_memory = []
        step_count = 0
        last_match_time = time.time()
        last_send_time = time.time()
        MATCH_INTERVAL = 0.3  # scene match는 0.3초마다 실행
        SEND_INTERVAL = 1.0  # 1초마다 전송
        

        
        while True:
            step_count += 1
            now = time.time()

            # 차량 제어 (20fps)
            front_vehicle.apply_control(front_agent.run_step())
            rear_vehicle.apply_control(rear_agent.run_step())

            if front_front_camera.frame is not None and (now - last_send_time >= SEND_INTERVAL):
                last_send_time = now

                # 이미지 전송
                front_sender.send_image(front_front_camera.frame, FRONT_CAMERA_TOPIC, timestamp=front_front_camera.timestamp)

                # 속도 전송
                velocity = front_vehicle.get_velocity()
                speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                front_sender.send_speed(speed)

                print(f"[SEND] 이미지 및 속도 전송됨 at {front_front_camera.timestamp:.2f}")

            # rear에서 front image 수신 → rear_memory에 저장
            if rear_receiver.front_image is not None:
                rear_memory.append({
                    'timestamp': rear_receiver.front_image_timestamp,
                    'front_image': rear_receiver.front_image.copy(),
                    'speed': rear_receiver.front_speed
                })
              
                # 최근 10초 이내의 메모리만 유지
                now_ts = get_current_timestamp()
                rear_memory = [m for m in rear_memory if now_ts - m['timestamp'] <= 10]

                # scene match는 일정 간격으로 실행
                if (now - last_match_time >= MATCH_INTERVAL and
                    rear_front_camera.frame is not None and
                    len(rear_memory) > 0):

                    last_match_time = now
                    live_img = rear_front_camera.frame.copy()

                    # 병렬 scene match 실행
                    futures = [executor.submit(scene_matcher.match_scene, m['front_image'], live_img) for m in rear_memory]
                    
                    results = []
                    for f in futures:
                        try:
                            results.append(f.result())
                        except Exception as e:
                            print(f"[ERROR] scene match 실패: {e}")
                            results.append(0.0)

                    # results 유효성 검사
                    if results and len(results) == len(rear_memory):
                        best_idx = int(np.argmax(results))
                        best_ratio = results[best_idx]

                        # rear_memory 범위 안전 확인
                        if best_idx < len(rear_memory):
                            best_past = rear_memory[best_idx]

                            if best_ratio >= 0.9:
                                sim_str = f"{best_ratio:.2f}"
                                front_vis = scene_matcher.draw_bbox(best_past['front_image'])
                                rear_vis = scene_matcher.draw_bbox(live_img)

                                cv2.imwrite(os.path.join(save_dir, f"{step_count}_match_front_{sim_str}.jpg"), front_vis)
                                cv2.imwrite(os.path.join(save_dir, f"{step_count}_match_rear_{sim_str}.jpg"), rear_vis)
                                print(f"[MATCH SAVED] similarity {sim_str}")

                                # 거리 추정 및 제어
                                if best_past['timestamp'] is not None and hasattr(rear_front_camera, 'timestamp'):
                                    delta_time = rear_front_camera.timestamp - best_past['timestamp']
                                    if best_past['speed'] is not None:
                                        est_dist = calculate_distance(best_past['speed'], delta_time)
                                        print(f"[INFO] 추정 거리: {pretty_print_distance(est_dist)}")
                                        action = controller.decide_action(est_dist)
                                        throttle, brake = controller.calculate_throttle_brake(action)
                                        rear_vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=0.0))
                    else:
                        print("[WARNING] scene match 결과 없음 또는 rear_memory 불일치 → 결과 생략")

            time.sleep(0.05)  # 차량 제어는 계속 빠르게 유지 (20fps)
            

    
    except KeyboardInterrupt:
        print("\n사용자 종료")

    except Exception as e:
        print(f"[EXCEPTION] 예외 발생: {e}")

    finally:
        print("액터 정리 중...")
        for actor in actor_list:
            try:
                if actor is not None:
                    actor.destroy()
            except Exception as destroy_err:
                print(f"[DESTROY ERROR] 액터 제거 실패: {destroy_err}")
        print("시뮬레이션 종료")

if __name__ == '__main__':
    main()


