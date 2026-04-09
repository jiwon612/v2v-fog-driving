import sys
import time
import carla
import math
import os
import cv2
import csv
from datetime import datetime
from carla import WeatherParameters

from sensors.camera_manager import CameraManager

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

    # 실험 환경 태그 설정
    collection_tag = 'clear'

    # 날짜별 디렉토리 생성
    today_str = datetime.now().strftime("%Y-%m-%d")
    front_base_dir = os.path.join("./yolo_dataset", collection_tag)
    front_dir = os.path.join(front_base_dir, today_str)
    os.makedirs(front_dir, exist_ok=True)

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        world = client.load_world('Town04')
        blueprint_library = world.get_blueprint_library()

        # 날씨 설정    
        if collection_tag == 'fog':
            weather = world.get_weather()
            weather.fog_density = 60.0
            weather.fog_distance = 0.0
            weather.fog_falloff = 1.0
            weather.precipitation = 0.0
            weather.wetness = 0.0
            weather.sun_altitude_angle = 30.0
            world.set_weather(weather)
            print("[WEATHER] 중간 안개 설정 완료")
        
        elif collection_tag == 'fog_heavy':
            weather = world.get_weather()
            weather.fog_density = 90.0
            weather.fog_distance = 5.0
            weather.fog_falloff = 1.0
            weather.precipitation = 0.0
            weather.wetness = 0.0
            weather.sun_altitude_angle = 10.0
            world.set_weather(weather)
            print("[WEATHER] 짙은 안개 설정 완료")
        
        elif collection_tag == 'clear':
            weather = WeatherParameters.ClearNoon
            weather.sun_altitude_angle = 70.0
            weather.sun_azimuth_angle=45.0
            weather.cloudiness = 10.0
            weather.wind_intensity=5.0
            world.set_weather(weather)
            print("[WEATHER] 맑은 하늘 적용 완료")
     
        spawn_props(world, blueprint_library)

        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        vehicle_bp.set_attribute('color', '255,255,255')  # 흰색 
        rear_vehicle = spawn_vehicle(world, vehicle_bp, 145.74, 11.27, 0.3)
        front_vehicle = spawn_vehicle(world, vehicle_bp, 105.74, 10.46, 0.3)
        front2_vehicle = spawn_vehicle(world, vehicle_bp, 125.74, 7.46, 0.3)

        actor_list.extend([front_vehicle, rear_vehicle, front2_vehicle])

        front_agent = BasicAgent(front_vehicle)
        rear_agent = BasicAgent(rear_vehicle)
        front2_agent = BasicAgent(front2_vehicle)

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

        front2_start_wp = map.get_waypoint(front2_vehicle.get_location())
        front2_route = []
        current_wp = front2_start_wp
        for _ in range(120):
            front2_route.append(current_wp)
            next_wps = current_wp.next(5.0)
            if not next_wps:
                break
            current_wp = next_wps[0]
        front2_agent.set_global_plan([(wp, RoadOption.LANEFOLLOW) for wp in front2_route])

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
        front2_agent.set_target_speed(55)

        rear_front_camera = CameraManager(world, rear_vehicle, carla.Transform(carla.Location(x=1.5, z=2.4)))

        step_count = 0

        while True:
            step_count += 1
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            if rear_front_camera.frame is not None and rear_front_camera.frame.mean() > 10:
                filename = f"front_{timestamp_str}.jpg"
                filepath = os.path.join(front_dir, filename)
                cv2.imwrite(filepath, rear_front_camera.frame)
                print(f"[SAVE] {filename}")

            front_vehicle.apply_control(front_agent.run_step())
            front2_vehicle.apply_control(front2_agent.run_step())
            rear_vehicle.apply_control(front_agent.run_step())
            
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[EXIT] 사용자 종료")

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