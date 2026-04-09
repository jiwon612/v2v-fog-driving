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

from math import sqrt
from sensors.camera_manager import CameraManager
from detection.car_detector import CarDetector
from config.settings import *

sys.path.append('/home/guest/carla-simulator/PythonAPI/carla')
sys.path.append('/home/guest/carla-simulator/PythonAPI')
from agents.navigation.basic_agent import BasicAgent
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
            transform = carla.Transform(location, carla.Rotation(yaw=270.0))
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
    save_dir = f"./car_detection/{timestamp_str}"
    os.makedirs(save_dir, exist_ok=True)

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        world = client.load_world('Town04')
        blueprint_library = world.get_blueprint_library()

        weather = world.get_weather()
        weather.fog_density = 60.0
        weather.fog_distance = 0.0
        weather.fog_falloff = 1.0
        weather.precipitation = 0.0
        weather.wetness = 0.0
        weather.sun_altitude_angle = 30.0
        world.set_weather(weather)
        print("[WEATHER] 짙은 안개만 설정 완료 (비 없음, 밝은 배경)")

        spawn_props(world, blueprint_library)

        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        vehicle_bp.set_attribute('color', '255,255,255')  # 흰색 
        rear_vehicle = spawn_vehicle(world, vehicle_bp, 245.74, 11.27, 0.3)
        front_vehicle = spawn_vehicle(world, vehicle_bp, 145.74, 10.46, 0.3)

        actor_list.extend([front_vehicle, rear_vehicle])

        front_agent = BasicAgent(front_vehicle)
        rear_agent = BasicAgent(rear_vehicle)

        map = world.get_map()
        start_wp = map.get_waypoint(front_vehicle.get_location())
        route = []
        current_wp = start_wp
        for _ in range(200):
            route.append(current_wp)
            next_wps = current_wp.next(5.0)
            if not next_wps:
                break
            current_wp = next_wps[0]

        front_agent.set_global_plan([(wp, RoadOption.LANEFOLLOW) for wp in route])

        rear_start_wp = map.get_waypoint(rear_vehicle.get_location())
        rear_route = []
        current_wp = rear_start_wp
        for _ in range(200):
            rear_route.append(current_wp)
            next_wps = current_wp.next(5.0)
            if not next_wps:
                break
            current_wp = next_wps[0]
        rear_agent.set_global_plan([(wp, RoadOption.LANEFOLLOW) for wp in rear_route])

        front_agent.set_target_speed(50)
        rear_agent.set_target_speed(70)
        rear_agent.ignore_vehicles(True)

        rear_front_camera = CameraManager(world, rear_vehicle, carla.Transform(carla.Location(x=1.5, z=2.4)))
        car_detector = CarDetector(model_path='./yolov5s.pt', device='cuda:3')

        step_count = 0
        override_timer = 0

        # rear_front_vedio 800x600 해상도, 20fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(save_dir, "rear_front_camera.mp4")
        video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (800, 600))

        while True:
            step_count += 1
            control = rear_agent.run_step()
         
            # rear_front_vedio 프레임 저장 
            if (rear_front_camera.frame is not None):
                video_writer.write(rear_front_camera.frame)
                
            # V2V 인식 및 제어 (1초 마다)
            if step_count % 20 == 0 and rear_front_camera.frame is not None:
                rear_live = rear_front_camera.frame
                
                rear_path = os.path.join(save_dir, f"{step_count}_match.jpg")
                detected, output_img = car_detector.detect_and_draw(rear_live, save_path=rear_path)
                
                # 제어
                if detected is True:
                    override_timer = 10 # 0.5초 
                    throttle, brake = 0.0, 0.3
                    control = carla.VehicleControl(throttle=throttle, brake=brake)

                    front_loc = front_vehicle.get_location()                                                            
                    rear_loc = rear_vehicle.get_location()                                                              
                                                                                                                                
                    distance_m = sqrt(                                                                                  
                    (front_loc.x - rear_loc.x) ** 2 +                                                               
                    (front_loc.y - rear_loc.y) ** 2 +                                                               
                    (front_loc.z - rear_loc.z) ** 2                                                                 
                    )                                                                                                   
                                                                                                                                
                    # km 단위로 변환                                                                                    
                    distance_km = distance_m / 1000
                    print(f"[DISTANCE] Distance between front and rear: {distance_m:.2f} meters ({distance_km:.4f} km)")
                                                                                                                                                                            
            elif (override_timer > 0):
                throttle, brake = 0.0, 0.3
                override_timer -= 1
                if throttle is not None and brake is not None:
                    control.throttle = throttle
                    control.brake = brake
                    control.steer = control.steer
                    print(f"[OVERRIDE ACTIVE] throttle={throttle}, brake={brake}")
                else:
                    print("[OVERRIDE ACTIVE] agent 주행 유지")
            else:
                # 뒷차 control
                control = rear_agent.run_step()
            # 앞차 제어 
            front_vehicle.apply_control(front_agent.run_step())
            # 뒷차 제어 
            rear_vehicle.apply_control(control)

            # simulation 종료 
            if front_agent.done():
                print("The target has been reached, stopping the simulation")
                break
            
            # 루프 간격 유지 (20fps)
            time.sleep(0.05)

        # rear_front_video 저장 
        video_writer.release()
    
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


