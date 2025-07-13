

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import time
import numpy as np
from keyboard_input_only_rev00 import controller #  키보드 이벤트 제어
#import win32ui
#import win32con
#import win32gui


import pydirectinput
import pydirectinput as pyautogui
pydirectinput.FAILSAFE = False
import torch
from keyboard_input_only_rev00 import controller

from yolopv2_inference.utils.utils import letterbox, split_for_trace_model, non_max_suppression, \
    driving_area_mask, lane_line_mask

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"E:/gta5_projectOCRtesseract/tesseract.exe"
sys.path.append(os.path.abspath("E:/gta5_project/AI_GTA5_Lanenet_Yolov2_Version/lanenet_inference"))

from yolopv2_inference.utils.utils import plot_one_box, clip_coords  # 추가되어 있어야 함, 객체결과표시 bbox기능



# Conclude setting / general reprocessing / plots / metrices / datasets
#yolopv2_inference.utils.utils --> 내가 수정한 경로
from yolopv2_inference.utils.utils import \
    time_synchronized,select_device, increment_path,\
    scale_coords,xyxy2xywh,non_max_suppression,split_for_trace_model,\
    driving_area_mask,lane_line_mask,plot_one_box,show_seg_result,\
    AverageMeter,\
    LoadImages

from preprocess_for_yolopv2 import grab_screen
from preprocess_for_yolopv2 import grab_speed_region
from virtual_lane import draw_virtual_centerline 

# YOLOPv2 모델 경로
WEIGHTS_PATH = "E:/gta5_project/AI_CAR_in_GTA5_with_YOLOPv2/yolopv2_inference/data/weights/yolopv2.pt"
IMGSZ = 640 # 입력 이미지 사이즈


# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 사용
model = torch.jit.load(WEIGHTS_PATH).to(device) # YOLOPv2 모델 로드
model.half()   # FP16 연산
model.eval()

# GPU warm-up
# GPU 워밍업 (빈 입력으로 한번 실행)
model(torch.zeros(1, 3, IMGSZ, IMGSZ).to(device).half())



#전역변수 초기화

last_known_speed = 0 # 전역 변수로 마지막 속도 저장
frame_count = 0

# 전역 상태
handle_angle = 0.0  # 현재 핸들 각도 (라디안)
handle_angle_zero = 0.0  # H 키 눌렀을 때의 기준 각도
handle_angle_initialized = False

# 키보드 처리 상태
auto_mode = False

# PID 제어 계수
KP = 1.0




###여기는 YoloPv2의 결과로부터 차산값을 받는 함수#################
# === YOLOPv2 inference ===
def get_segmentation_masks(model, img, device):
    model.eval()
    img_resized = letterbox(img, new_shape=640)[0]
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device).half()

    with torch.no_grad():
        output = model(img_tensor)

    if isinstance(output, tuple):
        drivable_output, lane_output = output[:2]
        if isinstance(drivable_output, list):
            drivable_output = drivable_output[0]
        if isinstance(lane_output, list):
            lane_output = lane_output[0]

        da_seg = drivable_output.argmax(0).byte().cpu().numpy()
        ll_seg = lane_output.argmax(0).byte().cpu().numpy()
    else:
        raise TypeError(f"[ERROR] Unexpected output type: {type(output)}")

    da_seg = cv2.resize(da_seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    ll_seg = cv2.resize(ll_seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return da_seg, ll_seg





##################여기서부터 핸들 각도 관련 함수들 ###############




# === Extract virtual centerline ===
#양쪽차선에서 가상 중심차선 추출
def extract_centerline_from_mask(ll_mask):
    h, w = ll_mask.shape
    left_points, right_points = [], []
    scan_rows = range(h - 1, int(h * 0.6), -10)
    for y in scan_rows:
        row = ll_mask[y, :]
        white_indices = np.where(row > 0)[0]
        if len(white_indices) > 10:
            mid = len(white_indices) // 2
            left_x = np.mean(white_indices[:mid])
            right_x = np.mean(white_indices[mid:])
            left_points.append((int(left_x), y))
            right_points.append((int(right_x), y))

    if left_points and right_points:
        centerline = [((l[0] + r[0]) // 2, l[1]) for l, r in zip(left_points, right_points)]
    elif left_points:
        centerline = [(x + 80, y) for x, y in left_points]
    elif right_points:
        centerline = [(x - 80, y) for x, y in right_points]
    else:
        centerline = []
    return centerline




# === Angle calculation ===
# 차선으로부터 추출된 가상 중앙처선과 무엇의 앵글을 비교?
def calculate_angle_from_centerline(centerline_points):
    if len(centerline_points) < 10:
        return 0.0
    head = np.mean(centerline_points[:5], axis=0)
    tail = np.mean(centerline_points[-5:], axis=0)
    dx = tail[0] - head[0]
    dy = tail[1] - head[1]
    angle = np.arctan2(dy, dx)
    return angle


#핸들 각도 상태 업데이트
def update_handle_angle(steering):
    global handle_angle
    handle_angle += steering  # 누적 업데이트
    handle_angle = np.clip(handle_angle, -math.pi / 4, math.pi / 4)  # -45도 ~ 45도 제한

#차선의 가상 중앙선
#차선으로부터의 추출된 중앙선인데...
def calculate_angle_from_centerline(centerline):
    if len(centerline) < 2:
        return 0.0

    # 앞쪽과 뒤쪽 평균으로 벡터 생성
    head = np.mean(centerline[:5], axis=0)
    tail = np.mean(centerline[-5:], axis=0)
    dx = tail[0] - head[0]
    dy = tail[1] - head[1]

    angle = math.atan2(dy, dx)
    return angle


#H키를 눌러서 핸들의 직진성을 컴퓨터에게 알려주는 역할
#자율주행 시작 키 'Y'누르기 전에 실행
def draw_handle_virtual_line(frame, handle_angle):
    h, w, _ = frame.shape
    cx = w // 2
    cy = h - 50

    line_length = 100
    dx = int(line_length * math.sin(handle_angle))
    dy = int(line_length * math.cos(handle_angle))

    pt1 = (cx, cy)
    pt2 = (cx + dx, cy - dy)

    color = (255, 0, 255)  # 보라색
    thickness = 2
    cv2.line(frame, pt1, pt2, color, thickness)
    cv2.putText(frame, f"Handle angle: {math.degrees(handle_angle):.1f} deg", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    return frame


#############여기까지 핸들 각도 관련 함수들##############



#############여기서부터 control 관련 함수들####################
# === Control ===
#key control
def apply_control(offset, threshold=10, slow_down_zone=20, max_speed_kmh=60,slow_down=False):
    global frame_count
    frame_count += 1 # 프레임 카운트 누적

    if offset is None: # 차선 정보 없음 → 모든 키에서 손 뗌
        pyautogui.keyUp('w')
        pyautogui.keyUp('a')
        pyautogui.keyUp('d')
        return

    offset = np.clip(offset, -100, 100)  # offset 값 제한
    current_speed = get_current_speed_from_screen() # 현재 속도 확인
    print(f"[INFO] 현재 속도: {current_speed} km/h")

    """ 
    #감속코드
    if current_speed is not None and current_speed < max_speed_kmh and abs(offset) < slow_down_zone:
        if frame_count % 10 == 0:
            pyautogui.keyDown('w')
        elif frame_count % 10 == 3:
            pyautogui.keyUp('w')

    """

    #위의 코드보다 부드럽게 감속
    # 속도 제어: 최대 속도보다 느리면 가속, 넘으면 감속

    """ 
    #이 코드 대로면 정지상태에서는 출발하지 않을 가능성 높음
    if current_speed is not None:
        if current_speed < max_speed_kmh - 5:
            pyautogui.keyDown('w')
        elif current_speed >= max_speed_kmh:
            pyautogui.keyUp('w')
    else:
        pyautogui.keyUp('w')  # 속도 모를 땐 감속
    """

    # 개선안  current_speed가 None일 경우, 가속 유도
    if current_speed is None:
        pyautogui.keyDown('w')  # 정지 상태라도 일단 출발 시도
        print("[INFO] 속도 인식 실패 → 기본 가속")
        return 0



  # 조향 제어: offset 기준 좌/우 핸들 조작
    if offset < -threshold:
        pyautogui.keyDown('a')  # 왼쪽 핸들
        pyautogui.keyUp('d')
        time.sleep(0.05)
    elif offset > threshold:
        pyautogui.keyDown('d') # 오른쪽 핸들
        pyautogui.keyUp('a')
        time.sleep(0.05)
    else: 
        pyautogui.keyUp('a')  #핸들 놓기
        pyautogui.keyUp('d')
         
    # 앞차 감지되면 가속키 떼기     
    if slow_down:
        pyautogui.keyUp('w')
        print("[INFO] 앞차와 거리 좁음 → 감속")
        return current_speed #이 값을 main drive loop함수로 전달 ,# 속도 리턴 (디버깅용)



def pid_control(angle):
    steering = KP * angle
    steering = np.clip(steering, -1.0, 1.0)
    return steering


#이 함수는 용도가 뭐더라--> 명확하게 모름
def process_keyboard(key):
    global auto_mode, handle_angle_initialized, handle_angle_zero

    if key == ord('h'):
        handle_angle_zero = handle_angle
        handle_angle_initialized = True
        print("[INFO] 핸들 정렬됨 (0도로 초기화됨)")

    elif key == ord('y') and handle_angle_initialized:
        auto_mode = True
        print("[INFO] 자율주행 모드 시작")

    elif key == ord('n'):
        auto_mode = False
        print("[INFO] 자율주행 중지")

    return auto_mode


######################여기까지 control 관련 함수들##################













# === Main ===
def main():
    global handle_angle, handle_angle_zero, handle_initialized, prev_steer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_yolopv2_model(device)

    print("[INFO] Press H to align handle, then Y to start driving...")
    driving = False

    while True:
        frame = grab_screen(region=(0, 0, 1280, 720))
        frame = cv2.resize(frame, (640, 480))

        if keyboard.is_pressed('h'):
            handle_angle = 0.0
            handle_angle_zero = 0.0
            handle_initialized = True
            print("[INFO] Handle aligned. Now press Y to start.")
            time.sleep(1)

        if keyboard.is_pressed('y') and handle_initialized:
            driving = True
            print("[INFO] Autonomous driving started.")
            time.sleep(1)

        if not driving:
            draw_handle_virtual_line(frame, handle_angle)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        da_mask, ll_mask = get_segmentation_masks(model, frame, device)
        centerline = extract_centerline_from_mask(ll_mask)

        for pt in centerline:
            cv2.circle(frame, pt, 2, (0, 255, 0), -1)

        angle = calculate_angle_from_centerline(centerline)
        steer = max(min(angle * 1.5, 0.5), -0.5)
        steer = 0.8 * prev_steer + 0.2 * steer
        prev_steer = steer

        apply_control(steer)
        draw_handle_virtual_line(frame, steer)

        cv2.imshow("Frame", frame)
        cv2.imshow("ll_mask", ll_mask * 255)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()



# 아래는 메인 루프에서 예시적으로 사용하는 코드입니다:
# frame = grab_screen()
# key = cv2.waitKey(1) & 0xFF
# auto_mode = process_keyboard(key)
# if auto_mode:
#     angle = calculate_angle_from_centerline(centerline)
#     steering = pid_control(angle)
#     update_handle_angle(steering)
#     apply_control(steering)
# frame = draw_handle_virtual_line(frame, handle_angle)
# cv2.imshow("Result", frame)
