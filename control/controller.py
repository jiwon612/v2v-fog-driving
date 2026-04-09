# ====== controller.py ======
class Controller:
    def __init__(self, safe_distance=70.0, min_distance=65.0, max_distance=75.0):
        self.safe_distance = safe_distance
        self.min_distance = min_distance
        self.max_distance = max_distance

        # 이전 상태를 기억하여 점진적 변화 가능
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        self.max_delta = 0.05  # 한 번에 바뀔 수 있는 최대 변화량

    def decide_action(self, estimated_distance):
        if estimated_distance < 60:
            return 'emergency'
        elif estimated_distance < self.min_distance:
            return 'decelerate'
        elif estimated_distance > self.max_distance:
            return 'accelerate'
        else:
            return 'maintain'

    def smooth_value(self, prev, target):
        delta = target - prev
        if abs(delta) > self.max_delta:
            delta = self.max_delta if delta > 0 else -self.max_delta
        return prev + delta
    
    def lane_change_steer_left(self):
        throttle = 0.0
        brake = 0.2
        steer = -0.25
        return throttle, brake, steer
        
    def lane_change_steer_right(self):
        throttle = 0.0
        brake = 0.2
        steer = 0.25
        return throttle, brake, steer

    def calculate_throttle_brake(self, action, estimated_distance):
        if action == 'maintain':
            print("[CONTROL] action: MAINTAIN → agent 주행 유지")

        elif action == 'accelerate':
            diff = estimated_distance - self.max_distance
            target_throttle = min(0.6, 0.2 + diff * 0.005)
            target_brake = 0.0
            print("[CONTROL] action: ACCELERATE → agent 주행 가속")

        elif action == 'decelerate':
            diff = self.min_distance - estimated_distance
            target_throttle = 0.0
            target_brake = min(0.6, 0.2 + diff * 0.01)
            print("[CONTROL] action: DECELERATE → agent 주행 감속")

        elif action == 'emergency':
            print("[CONTROL] action: EMERGENCY → 차선 변경 우선, 감속 제어 생략")
            
        # 점진적으로 변화
        throttle = self.smooth_value(self.prev_throttle, target_throttle)
        brake = self.smooth_value(self.prev_brake, target_brake)

        # brake와 throttle 동시에 들어가지 않도록 강제
        if throttle > 0.01:
            brake = 0.0
        elif brake > 0.01:
            throttle = 0.0

        # 상태 저장
        self.prev_throttle = throttle
        self.prev_brake = brake

        print(f"[CONTROL] action: {action.upper():<10} → throttle: {throttle:.2f}, brake: {brake:.2f}")
        return throttle, brake
