# drive_with_yolopv2_control_rev25.py

import cv2
import numpy as np
import torch
import pyautogui
import time
import math
from preprocess_for_yolopv2 import letterbox, grab_screen
from preprocess_for_yolopv2 import grab_screen, letterbox
import time
import keyboard
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
# Conclude setting / general reprocessing / plots / metrices / datasets
#yolopv2_inference.utils.utils --> 내가 수정한 경로
from yolopv2_inference.utils.utils import \
    time_synchronized,select_device, increment_path,\
    scale_coords,xyxy2xywh,non_max_suppression,split_for_trace_model,\
    driving_area_mask,lane_line_mask,plot_one_box,show_seg_result,\
    AverageMeter,\
    LoadImages

from yolopv2_inference.utils.utils import plot_one_box, clip_coords  # 추가되어 있어야 함, 객체결과표시 bbox기능

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"E:/gta5_projectOCRtesseract/tesseract.exe"
sys.path.append(os.path.abspath("E:/gta5_project/AI_GTA5_Lanenet_Yolov2_Version/lanenet_inference"))


from preprocess_for_yolopv2 import grab_screen
from preprocess_for_yolopv2 import grab_speed_region #이거 속도계 영역
#from virtual_lane import draw_virtual_centerline 

# YOLOPv2 모델 경로
WEIGHTS_PATH = "E:/gta5_project/AI_CAR_in_GTA5_with_YOLOPv2/yolopv2_inference/data/weights/yolopv2.pt"
IMGSZ = 640 # 입력 이미지 사이즈 ,  # YOLOPv2 입력 이미지 사이즈 (기본값 640x640)
# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 사용
model = torch.jit.load(WEIGHTS_PATH).to(device) # YOLOPv2 모델 로드
model.half()   # FP16 연산
model.eval()

# GPU warm-up
# GPU 워밍업 (빈 입력으로 한번 실행)
model(torch.zeros(1, 3, IMGSZ, IMGSZ).to(device).half())


from models.common import DetectMultiBackend  #이걸 왜 main함수 내에서 로딩함?
import torch.backends.cudnn as cudnn #이걸 왜 main함수 내에서 로딩함?

#전역변수 초기화 ---> 버전 20이전부터
#last_known_speed = 0 # 전역 변수로 마지막 속도 저장

####여기까지 내가 기존에 쓰던 모듈로딩#####


# === 전역 변수 === 버전 25부터
handle_angle_zero = None  # H키 누를 때의 핸들 각도 기준
handle_angle = 0.0        # 현재 가상 핸들 각도
auto_mode = False         # Y키로 자율주행 시작 여부
prev_steer = 0.0          # 저역 필터용

frame_count = 0  # apply_control() 외부에서 선언 필요


# === 누락된 함수 정의 ===
def load_yolopv2_model(device):
    model = torch.jit.load("best.torchscript.pt")
    model.to(device).eval()
    return model

def get_current_speed_from_screen():
    # 임시 속도 반환 (OCR 기능 미구현 상태)
    return 30.0

# === 시각화: 핸들 기준 가상선 ===
def draw_handle_virtual_line(frame, handle_angle):
    h, w = frame.shape[:2]
    cx, cy = w // 2, int(h * 0.9)
    length = 100
    angle_rad = math.radians(handle_angle)
    dx = int(length * math.sin(angle_rad))
    dy = int(length * math.cos(angle_rad))
    cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy - dy), (0, 255, 255), 2)

# === 중심선 추출 ===

""" 
# === 차선에서 가상 중심선 추출 ===
def extract_centerline_from_mask(ll_mask):
    height, width = ll_mask.shape
    centerline = []

    for y in range(height - 1, height // 2, -10):  # 아래쪽 절반만 스캔
        row = ll_mask[y]
        left_x = np.argmax(row > 0)
        right_x = width - np.argmax(np.flip(row) > 0)

        if left_x == 0 or right_x == width:
            continue

        center_x = (left_x + right_x) // 2
        centerline.append((center_x, y))

    return centerline
"""
def extract_centerline_from_mask(mask):
    h, w = mask.shape
    left_points, right_points = [], []
    for y in range(h-1, int(h*0.6), -5):
        line = mask[y]
        nonzeros = np.nonzero(line)[0]
        if len(nonzeros) > 10:
            mid = w // 2
            left = nonzeros[nonzeros < mid]
            right = nonzeros[nonzeros >= mid]
            if len(left) > 0:
                left_points.append((int(np.mean(left)), y))
            if len(right) > 0:
                right_points.append((int(np.mean(right)), y))
    center_points = []
    for l, r in zip(left_points, right_points):
        cx = (l[0] + r[0]) // 2
        cy = (l[1] + r[1]) // 2
        center_points.append((cx, cy))
    return center_points



# === PID 조향 제어 ===

prev_steer = 0
def pid_control(angle, kp=1.0):
    global prev_steer
    steer = kp * angle
    steer = max(min(steer, 0.5), -0.5)
    steer = 0.8 * prev_steer + 0.2 * steer
    prev_steer = steer
    return steer




############여기까지 전반부 (에디터 1)################



##########여기서부터 후반부 (에디터2) ###############


# === YOLOPv2 세그멘테이션 결과 추출 ===
def get_segmentation_masks(model, img, device):
    model.eval()

    img_resized = letterbox(img, new_shape=640)[0]
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device).half()

    with torch.no_grad():
        output = model(img_tensor)

    # 출력 형태에 따라 분기 처리
    if isinstance(output, dict):
        da_seg = output['drivable'][0].argmax(0).byte().cpu().numpy()
        ll_seg = output['lane'][0].argmax(0).byte().cpu().numpy()
    elif isinstance(output, (list, tuple)):
        drivable_output = output[0][0] if isinstance(output[0], list) else output[0]
        lane_output = output[1][0] if isinstance(output[1], list) else output[1]
        da_seg = drivable_output.argmax(0).byte().cpu().numpy()
        ll_seg = lane_output.argmax(0).byte().cpu().numpy()
    else:
        raise TypeError(f"[ERROR] 예상치 못한 출력 타입: {type(output)}")

    da_seg = cv2.resize(da_seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    ll_seg = cv2.resize(ll_seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    return da_seg, ll_seg





# === 중심선으로부터 angle 계산 ===
def calculate_angle_from_centerline(centerline):
    if len(centerline) < 10:
        return 0  # 직진 유지

    head = np.mean(centerline[:5], axis=0)
    tail = np.mean(centerline[-5:], axis=0)
    dx = tail[0] - head[0]
    dy = tail[1] - head[1]
    angle = np.arctan2(dy, dx)
    return angle






# === 키보드 제어 입력 ===



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



    #부드럽게 감속
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



#자율주행중에  n눌러서 정지
def stop_control():
    pyautogui.keyUp('w')
    pyautogui.keyUp('a')
    pyautogui.keyUp('d')








# === 메인 루프 ===
def main(): 
    #from models.common import DetectMultiBackend  #이걸 왜 main함수 내에서 로딩함?
    #import torch.backends.cudnn as cudnn #이걸 왜 main함수 내에서 로딩함?

    #weights_path = './YOLOPv2/best.torchscript.pt' #이게 뭐임?? --> 잘못된 경로
   
    print("[INFO] 자율주행을 시작하려면 H키로 핸들 정렬 후 Y키를 누르세요.")
    handle_initialized = False #이게 while문안에 있으면 매frame마다 H눌러야함
    running = False
    


    while True:
        frame = grab_screen(region=(0, 0, 1280, 720)) # 풀 프레임 캡처
        
        vis_img = frame.copy() # 속도계 영역 시각화용 복사
        
        # === 속도계 영역 시각화 ===
        cv2.rectangle(vis_img, (1180, 660), (1280, 710), (255, 0, 0), 2)  # BGR, 두께 2
        # 이후 YOLO 처리
        frame = cv2.resize(frame, (640, 480))

        

        if handle_initialized:
            draw_handle_virtual_line(vis_img, handle_angle)


        if keyboard.is_pressed('h'):
            handle_initialized = True
            print("[INFO] 핸들 정렬 기준선 설정됨 (0도).")
            cv2.line(vis_img, (320, 480), (320, 240), (255, 255, 255), 2)
            time.sleep(0.5)

        if keyboard.is_pressed('y') and handle_initialized:
            running = True
            print("[INFO] 자율주행 시작!")
            time.sleep(0.5)

            if keyboard.is_pressed('n'):
                running = False
                stop_control()
                print("[INFO] 자율주행 중지됨")
                time.sleep(0.5)


        da_seg, ll_seg = get_segmentation_masks(model, frame, device)
        centerline = extract_centerline_from_mask(ll_seg)
        angle = calculate_angle_from_centerline(centerline)
        steer = pid_control(angle)

        for (x, y) in centerline:
            cv2.circle(vis_img, (int(x), int(y)), 2, (0, 255, 255), -1)

        #if handle_initialized:
         #   cv2.line(vis_img, (320, 480), (320, 240), (255, 255, 255), 2)

         # 제안 코드
        if handle_initialized:
            draw_handle_virtual_line(vis_img, 0)  # 0도 직선

        if running:
            apply_control(steer)

        cv2.imshow("YOLOPv2 Result", vis_img)
        cv2.imshow("Lane Mask", ll_seg * 255)
        cv2.moveWindow("YOLOPv2 Result", 1280, 100)
        cv2.moveWindow("Lane Mask", 1280, 600)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
