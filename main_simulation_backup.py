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

sys.path.append('/home/guest/carla-simulator/PythonAPI/carla')
sys.path.append('/home/guest/carla-simulator/PythonAPI')
from agents.navigation.behavior_agent import BehaviorAgent
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
    save_dir = f"./scene_match/{timestamp_str}"
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
        for _ in range(120):
            route.append(current_wp)
            next_wps = current_wp.next(5.0)
            if not next_wps:
                break
            current_wp = next_wps[0]

        front_agent.set_global_plan([(wp, RoadOption.LANEFOLLOW) for wp in route])

        rear_start_wp = map.get_waypoint(rear_vehicle.get_location())
        rear_route = []
        current_wp = rear_start_wp
        for _ in range(120):
            rear_route.append(current_wp)
            next_wps = current_wp.next(5.0)
            if not next_wps:
                break
            current_wp = next_wps[0]
        rear_agent.set_global_plan([(wp, RoadOption.LANEFOLLOW) for wp in rear_route])

        front_agent.set_target_speed(50)
        rear_agent.set_target_speed(60)

        front_front_camera = CameraManager(world, front_vehicle, carla.Transform(carla.Location(x=1.5, z=2.4)))
        rear_front_camera = CameraManager(world, rear_vehicle, carla.Transform(carla.Location(x=1.5, z=2.4)))

        front_sender = MQTTSender()
        front_sender.loop()

        rear_receiver = MQTTReceiver()
        rear_receiver.client.connect(MQTT_BROKER, MQTT_PORT, 60)
        rear_receiver.client.loop_start()

        scene_matcher = SceneMatcher(model_path=YOLO_MODEL_PATH, device='cuda:3')
        controller = Controller()

        rear_memory = []
        step_count = 0
        
        lane_change = 1;

        override_throttle = None
        override_brake = None
        override_timer = 0

        # rear_front_vedio 800x600 해상도, 20fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(save_dir, "rear_front_camera.mp4")
        video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (800, 600))

        while True:
            step_count += 1
            action = None
         
            # 전방 이미지 전송 및 속도 전송 
            if front_front_camera.frame is not None:
                front_sender.send_image(front_front_camera.frame, FRONT_CAMERA_TOPIC, timestamp=front_front_camera.timestamp)

                velocity = front_vehicle.get_velocity()
                speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                front_sender.send_speed(speed)
                
            # 3. rear에서 front image 수신 → rear_memory에 저장
            if rear_receiver.front_image is not None:
                now = get_current_timestamp()
                
                if not rear_memory or rear_receiver.front_image_timestamp - rear_memory[-1]['timestamp'] >= 0.5:
                    rear_memory.append({
                        'timestamp': rear_receiver.front_image_timestamp,
                        'front_image': rear_receiver.front_image.copy(),
                        'speed': rear_receiver.front_speed
                    })
                # 최근 5초 이내의 메모리 유지
                # 0.5초 마다 이미지 저장 
                if (step_count % 10 == 0):
                    rear_memory = [m for m in rear_memory if now - m['timestamp'] <= 5]
            
            # rear_front_vedio 프레임 저장 
            if (rear_front_camera.frame is not None):
                video_writer.write(rear_front_camera.frame)
                
            # V2V 인식 및 제어 (1초 마다)
            if step_count % 20 == 0 and rear_front_camera.frame is not None and rear_memory:
                rear_live = rear_front_camera.frame
                best_match, max_similarity = scene_matcher.match_scene_multiple(rear_memory, rear_live)

                if max_similarity >= 0.8:   
                    # 소수점 2자리로 유사도 포맷
                    sim_str = f"{max_similarity:.2f}"
                    
                    # bbox 그리기
                    front_vis = scene_matcher.draw_bbox(best_match['front_image'])
                    rear_vis = scene_matcher.draw_bbox(rear_live)

                    front_path = os.path.join(save_dir, f"{step_count}_match_front_{sim_str}.jpg")
                    rear_path = os.path.join(save_dir, f"{step_count}_match_rear_{sim_str}.jpg")
                    
                    cv2.imwrite(front_path, front_vis) 
                    cv2.imwrite(rear_path, rear_vis)
                    
                    print(f"[MATCH SAVED] front & rear images saved with similarity {sim_str}")
                    
                    # 거리 추정 및 제어
                    if best_match['timestamp'] is not None and hasattr(rear_front_camera, 'timestamp'):
                        delta_time = rear_front_camera.timestamp - best_match['timestamp']
                        if best_match['speed'] is not None:
                            estimated_distance = calculate_distance(best_match['speed'], delta_time)
                            print(f"[INFO] 추정 거리: {pretty_print_distance(estimated_distance)}")

                            action = controller.decide_action(estimated_distance)
                            lane_wp = map.get_waypoint(rear_vehicle.get_location())
                            
                            if (action == 'emergency') and (lane_wp.lane_id != -2):
                                print(f"[DEBUG] emergency 진입")
                                override_timer = 5 # 0.25초
                                
                                throttle, brake, steer = controller.lane_change_steer()
                                control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
                                
                            if action in ['decelerate', 'accelerate']:
                                override_timer = {
                                    'decelerate': 10, # 0.5초 
                                    'accelerate': 10  # 0.5초
                                }[action]
                                
                                controller.prev_throttle = control.throttle
                                controller.prev_brake = control.brake
                                
                                # 초기 throttle/brake 계산 + 로그
                                throttle, brake = controller.calculate_throttle_brake(action, estimated_distance)
                                control = carla.VehicleControl(throttle=throttle, brake=brake)
                                print(f"[OVERRIDE START] action={action.upper()}, throttle={throttle:.2f}, brake={brake:.2f}")

            elif (override_timer > 0):
                if action in ['decelerate', 'accelerate']:
                    throttle, brake = controller.calculate_throttle_brake(action, estimated_distance)
                elif action == 'emergency':
                    if (override_timer > 2):
                        throttle, brake, steer = controller.lane_change_steer()
                    else:
                        throttle, brake, steer = 0.4, 0.0, 0.0
                else:
                    throttle, brake = 0.0, 0.0  # emergency 등은 override 미적용
                override_timer -= 1
                if throttle is not None and brake is not None:
                    control.throttle = throttle
                    control.brake = brake
                    control.steer = control.steer
                    print(f"[OVERRIDE ACTIVE] throttle={control.throttle:.2f}, brake={control.brake:.2f}")
                else:
                    print("[OVERRIDE ACTIVE] maintain 상태 → agent 주행 유지")
            else:
                # 뒷차 control
                now_wp = map.get_waypoint(rear_vehicle.get_location())
                if (now_wp.lane_id == -2) and (lane_change == 1):
                    new_wp = map.get_waypoint(rear_vehicle.get_location(), lane_type=carla.LaneType.Driving)
                    # 2차선(global plan)에 맞는 route 재설정
                    rear_route = []
                    current_wp = new_wp
                    for _ in range(120):
                        rear_route.append(current_wp)
                        next_wps = current_wp.next(5.0)
                        if not next_wps:
                            break
                        current_wp = next_wps[0]
                    rear_agent.set_global_plan([(wp, RoadOption.LANEFOLLOW) for wp in rear_route])
                    
                    lane_change -= 1
                    print("차선 변경 완료, agent 자동주행 재개")
                # 여기서 global plan을 새 차선 기준으로 새로 만들어줘도 좋음!
                # (필요시 아래 참고)
                # 덮어쓰기 적용 중이면 계속 유지
                control = rear_agent.run_step()
            # 앞차 제어 
            front_vehicle.apply_control(front_agent.run_step())
            # 뒷차 제어 
            rear_vehicle.apply_control(control)
            # 뒷차 속도 프린트              
            rear_velocity = rear_vehicle.get_velocity()
            rear_speed = 3.6 * math.sqrt(rear_velocity.x**2 + rear_velocity.y**2 + rear_velocity.z**2)
            print(f"[REAR SPEED] {rear_speed:.2f} km/h")
            
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


