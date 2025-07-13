# drive_with_yolopv2_control_rev25.py

import cv2
import numpy as np
import torch
import pyautogui
import time
import math
from preprocess_for_yolopv2 import grab_screen
from yolopv2_inference.utils.utils import letterbox
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


#from models.common import DetectMultiBackend  #이걸 왜 main함수 내에서 로딩함?
#import torch.backends.cudnn as cudnn #이걸 왜 main함수 내에서 로딩함?

#전역변수 초기화 ---> 버전 20이전부터
#last_known_speed = 0 # 전역 변수로 마지막 속도 저장

####여기까지 내가 기존에 쓰던 모듈로딩#####


# === 전역 변수 === 버전 25부터
handle_angle_zero = None  # H키 누를 때의 핸들 각도 기준
handle_angle = 0.0        # 현재 가상 핸들 각도
handle_angle = max(min(handle_angle, 90), -90) #너무 오래 누적되면 ±90도 이상 각도도 나올 수 있으니, 이처럼 제한 걸 수도 있음
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
""" 
def draw_handle_virtual_line(frame, handle_angle):
    h, w = frame.shape[:2]
    cx, cy = w // 2, int(h * 0.9)
    length = 100
    angle_rad = math.radians(handle_angle)
    dx = int(length * math.sin(angle_rad))
    dy = int(length * math.cos(angle_rad))
    cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy - dy), (0, 255, 255), 2)
"""

#원근감 적용
def draw_handle_virtual_line(frame, handle_angle):
    h, w = frame.shape[:2]
    cx, cy = w // 2, int(h * 0.9)  # 화면 아래쪽 중심
    max_length = 100
    color = (0, 255, 255)  # 노란색

    # 여러 개의 짧은 선분을 그려 원근감을 표현
    for i in range(10, max_length + 1, 10):
        angle_rad = math.radians(handle_angle)
        dx1 = int((i - 10) * math.sin(angle_rad))
        dy1 = int((i - 10) * math.cos(angle_rad))
        dx2 = int(i * math.sin(angle_rad))
        dy2 = int(i * math.cos(angle_rad))

        thickness = max(1, int(i / 25))  # 가까울수록 두껍게
        cv2.line(frame, (cx + dx1, cy - dy1), (cx + dx2, cy - dy2), color, thickness)





# === 중심선 추출 ===


#원근감 있게 개선
def extract_centerline_from_mask(mask, output_img=None):
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

    if output_img is not None:
        # 좌우 차선 점 표시
        for x, y in left_points:
            cv2.circle(output_img, (x, y), 2, (255, 0, 0), -1)  # 파랑 = 왼쪽
        for x, y in right_points:
            cv2.circle(output_img, (x, y), 2, (0, 0, 255), -1)  # 빨강 = 오른쪽

        # 중심선 라인 연결 (원근감: 아래쪽은 두껍게)
        for i in range(1, len(center_points)):
            p1, p2 = center_points[i - 1], center_points[i]
            thickness = max(1, int((h - p1[1]) / 60))  # y값이 낮을수록 두껍게
            cv2.line(output_img, p1, p2, (0, 255, 255), thickness)  # 노란 중심선

    return center_points




# === PID 조향 제어 ===

prev_steer = 0


#차선 각도(B) - 핸들 상태(A)의 차이를 받아 조향 명령 계산
def pid_control(angle_error, kp=1.0):
    """
    조향 보정량(steering command) 계산 함수
    angle_error = (중심선 각도 B) - (현재 핸들 각도 A)
    출력값: steer_cmd (-1.0 ~ +1.0 사이)
    """
    steer = kp * angle_error
    steer = max(min(steer, 0.5), -0.5)  # 조향 제한
    return steer




############여기까지 전반부 (에디터 1)################



##########여기서부터 후반부 (에디터2) ###############


# === YOLOPv2 세그멘테이션 결과 추출 ===




def get_segmentation_masks(model, img, device):
    model.eval()

    # --- 이미지 전처리 ---
    img_resized = letterbox(img, new_shape=640)[0]
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device).half()

    with torch.no_grad():
        output = model(img_tensor)

    # === 내부 Tensor 추출 보조 함수 ===
    def unwrap_tensor(nested):
        """
        YOLOPv2 출력이 다음처럼 중첩되어 있는 경우:
        - [tensor]
        - [[tensor]]
        - [[[tensor]]]
        이럴 때, 내부의 tensor를 꺼낼 때까지 반복해서 벗겨내는 함수
        """
        while isinstance(nested, (list, tuple)):
            if len(nested) == 0:
                raise ValueError("[ERROR] YOLOPv2 출력이 비어 있음")
            nested = nested[0]
        if not isinstance(nested, torch.Tensor):
            raise TypeError(f"[ERROR] tensor가 아님: {type(nested)}")
        return nested

    # === 출력 분기 처리 ===
    if isinstance(output, dict):
        # dict 구조로 나올 경우
        da_seg = output['drivable'][0].argmax(0).byte().cpu().numpy()
        ll_seg = output['lane'][0].argmax(0).byte().cpu().numpy()

    elif isinstance(output, (list, tuple)):
        # list/tuple 구조로 나올 경우
        drivable_output = unwrap_tensor(output[0])  # ← 중첩 껍질 벗기기
        lane_output = unwrap_tensor(output[1])

        # drivable / lane 마스크 추출 (0: 배경, 1: 가능, 2: 차선)
        da_seg = drivable_output.argmax(0).byte().cpu().numpy()
        ll_seg = lane_output.argmax(0).byte().cpu().numpy() #이 결과가 비어있을 가능성 --> cv2.error: ... resize.cpp:3845: error: (-215:Assertion failed) !dsize.empty()
                                                            #왜 비어있을까? 입력된 이미지가 어두움, 게임 내 차선이 흐릿함 -->  YOLOPv2가 lane을 검출 못함 또는 모델이 GPU로 잘 로드되지 않아서 출력이 무효값

    else:
        raise TypeError(f"[ERROR] 예상치 못한 출력 타입: {type(output)}")

    # === YOLOPv2 출력 → 원본 이미지 크기로 리사이즈 ===
    da_seg = cv2.resize(da_seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # === 에러 방지: 비어있을 경우 예외 발생 ===
    if ll_seg is None or ll_seg.size == 0:
        raise ValueError("[ERROR] YOLOPv2 lane segmentation 결과가 비어 있음") 
    if ll_seg is None or ll_seg.size == 0:
        print("[WARNING] YOLOPv2 lane 출력이 비어 있음. 현재 화면 상태 확인 요망.")
        return None, None  # 또는 ll_seg = np.zeros_like(img[..., 0])

    ll_seg = cv2.resize(ll_seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    return da_seg, ll_seg






# === 중심선으로부터 angle 계산 ===
#가상 중심선(centerline)의 방향 벡터를 계산해서 조향 각도로 변환
#차량이 따라가야 할 가상의 중심선을 구성한 후, 이 선의 방향이 화면의 수직선(차량 기준 정면)과 얼마나 다른지를 각도로 계산
#이 값이 조향(steering)의 기준 angle





#핸들 기준선 0도에 대해 중심선 각도(B)를 계산하는 함수
def calculate_angle_from_centerline(centerline):
    """
    중심선 각도(B)를 계산하는 함수.
    단위: 라디안
    """
    global handle_angle_zero

    if len(centerline) < 10:
        return 0  # 데이터 부족 시 조향 없음

    # 중심선 방향 벡터 계산
    head = np.mean(centerline[:5], axis=0)
    tail = np.mean(centerline[-5:], axis=0)
    dx = tail[0] - head[0]
    dy = tail[1] - head[1]
    centerline_angle = np.arctan2(dy, dx)  # 라디안 단위

    # 핸들 기준선 0도 설정 (H키 눌렀을 때)
    if handle_angle_zero is None:
        handle_angle_zero = centerline_angle
        print(f"[INFO] 핸들 기준선(0도) 설정됨: {math.degrees(handle_angle_zero):.2f}°")

    # 중심선의 현재 방향(B) = 기준선 대비 기울기
    angle_to_lane = centerline_angle - handle_angle_zero
    return angle_to_lane



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


#ll_seg는 반드시 if running: 안에서만 사용되며, None 또는 size == 0 검사로 방어
#cv2.imshow("Lane Mask", ...)도 if running: 안에만 존재
#handle_angle은 전역 변수로 누적 관리
#draw_handle_virtual_line()은 핸들 초기화 시 정상 호출
#시각화(vis_img)는 항상 유지되며, 오류 발생 시에도 YOLOPv2 Result는 보여짐


def main():
    global handle_angle
    handle_initialized = False
    running = False

    print("[INFO] 자율주행을 시작하려면 H키로 핸들 정렬 후 Y키를 누르세요.")

    while True:
        screen = grab_screen()
        if screen is None:
            print("[ERROR] 화면 캡처 실패.")
            break

        #frame = cv2.resize(screen, (640, 480)) 
        #frame = screen.copy()  # 원본을 그대로 사용
        #vis_img = cv2.resize(screen, (1280, 720))    # 시각화용
        frame = cv2.resize(screen, (1280, 720)) #모델입력 , 여기서 사이즈를 지정안하면, 다른 사이즈로 출력되는듯하여 이렇게 지정
        vis_img = frame.copy() # 시각화용 ,'()' 괄호가 빠지면 안됨, 함수 호출로 바꿔야 이미지 복사가 되기 때문

        # === 속도계 영역 시각화 ===
        #cv2.rectangle(vis_img, (1180, 660), (1280, 710), (255, 0, 0), 2)

        # === 핸들 기준선 시각화 ===
        if handle_initialized:
            draw_handle_virtual_line(vis_img, handle_angle)

        # === 핸들 정렬 (H키) ===
        if keyboard.is_pressed('h'):
            handle_initialized = True
            handle_angle = 0
            print("[INFO] 핸들 정렬 기준선 설정됨 (0도).")
            time.sleep(0.5)

        # === 자율주행 시작 (Y키) ===
        if keyboard.is_pressed('y') and handle_initialized:
            running = True
            print("[INFO] 자율주행 시작!")
            time.sleep(0.5)

        # === 자율주행 중지 (N키) ===
        if keyboard.is_pressed('n'):
            running = False
            stop_control()
            print("[INFO] 자율주행 중지됨")
            time.sleep(0.5)

        # === 자율주행 로직 ===
        if running:
            try:
                da_seg, ll_seg = get_segmentation_masks(model, frame, device)
                if da_seg is None or ll_seg is None:
                    print("[WARNING] YOLO 출력 비정상 → 건너뜀")
                    continue

                centerline = extract_centerline_from_mask(ll_seg, vis_img)
                for (x, y) in centerline:
                    cv2.circle(vis_img, (int(x), int(y)), 2, (0, 255, 255), -1)

                angle_to_lane = calculate_angle_from_centerline(centerline)
                angle_error = angle_to_lane - handle_angle
                steer = pid_control(angle_error)

                apply_control(steer)
                handle_angle += steer * 5.0  # 누적 핸들 업데이트
                handle_angle = max(min(handle_angle, math.radians(90)), math.radians(-90))

                print(f"[DEBUG] 조향입력: {math.degrees(steer):.2f}°, 누적핸들: {math.degrees(handle_angle):.2f}°")

                # === 차선 마스크 시각화 ===
                # ll_seg가 있을 때만 표시 가능 → if running: 블록 안에서만 안전하게 사용 가능
                cv2.imshow("Lane Mask", ll_seg * 255)
                cv2.resizeWindow("Lane Mask", 640, 360)
                cv2.moveWindow("Lane Mask", 1280, 100)

            except Exception as e:
                print(f"[ERROR] 자율주행 중 예외 발생: {e}")
                continue

        # === 전체 시각화 ===
        # 화면 시각화를 항상 유지하는 게 목적 → 자율주행 중이 아니더라도 화면을 사용자에게 보여주기 위해 바깥에 위치
        cv2.imshow("YOLOPv2 Result", vis_img)
        cv2.resizeWindow("YOLOPv2 Result", 640, 360)
        cv2.moveWindow("YOLOPv2 Result", 1280, 600)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()




""" 
#rev02

def main(): 
    print("[INFO] 자율주행을 시작하려면 H키로 핸들 정렬 후 Y키를 누르세요.")
    handle_initialized = False
    running = False
    global handle_angle

    while True:
        screen = grab_screen()
        frame = cv2.resize(screen, (640, 480))
        vis_img = cv2.resize(screen, (1280, 720))

        # === 속도계 영역 시각화 ===
        cv2.rectangle(vis_img, (1180, 660), (1280, 710), (255, 0, 0), 2)

        # === 핸들 기준선 설정 ===
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

        # === 자율주행 루프 ===
        if running:
            da_seg, ll_seg = get_segmentation_masks(model, frame, device)
            if da_seg is None or ll_seg is None:
                print("[WARNING] YOLO 출력 비정상 → 건너뜀")
                continue

            centerline = extract_centerline_from_mask(ll_seg, vis_img)
            for (x, y) in centerline:
                cv2.circle(vis_img, (int(x), int(y)), 2, (0, 255, 255), -1)

            angle_to_lane = calculate_angle_from_centerline(centerline)
            angle_error = angle_to_lane - handle_angle
            steer = pid_control(angle_error)

            apply_control(steer)
            handle_angle += steer * 5.0  # 누적 핸들 업데이트
            handle_angle = max(min(handle_angle, math.radians(90)), math.radians(-90))

            print(f"[DEBUG] 조향입력: {math.degrees(steer):.2f}°, 누적핸들: {math.degrees(handle_angle):.2f}°")

            cv2.imshow("Lane Mask", ll_seg * 255)
            cv2.resizeWindow("Lane Mask", 640, 360)
            cv2.moveWindow("Lane Mask", 1280, 100)

        cv2.imshow("YOLOPv2 Result", vis_img)
        cv2.resizeWindow("YOLOPv2 Result", 640, 360)
        cv2.moveWindow("YOLOPv2 Result", 1280, 600)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

"""




""" 
#rev01
def main(): 
  

   
    print("[INFO] 자율주행을 시작하려면 H키로 핸들 정렬 후 Y키를 누르세요.")
    handle_initialized = False #이게 while문안에 있으면 매frame마다 H눌러야함
    running = False
    


    while True:
        screen = grab_screen() # 풀 프레임 캡처
        frame = cv2.resize(screen, (640, 480))  # YOLO 추론용
        vis_img = cv2.resize(screen, (1280,720))  # 시각화용
        #vis_img = frame.copy() # 속도계 영역 시각화용 복사,  지금은 의미 없음
        #vis_img = screen.copy() #이렇게 해도 ok라는 (속도계 영역 시각화)

        # 자율주행 모드일 경우만 YOLO 실행
        if running:
            da_seg, ll_seg = get_segmentation_masks(model, frame, device)

             # [여기에 삽입]
            if da_seg is None or ll_seg is None:
                print("[WARNING] YOLO 출력이 비어 있어 건너뜀.")
                continue

        
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


       
        centerline = extract_centerline_from_mask(ll_seg, vis_img)  #여기에 vis_img 넘기면 자동 시각화 
                                                                    #이 코드에서 'UnboundLocalError: local variable 'll_seg' referenced before assignment' 에러발생
                                                                    #ll_seg는 자율주행이 시작된 이후에만 생성됨, if running:  da_seg, ll_seg = get_segmentation_masks(model, frame, device)
                                                                    #이 코드 블록이 실행되지 않은 상태에서 ll_seg를 사용하는 코드는 실행되면 안 됨.
                                                                    #extract_centerline_from_mask(ll_seg, vis_img) 이 호출되는 부분을 반드시 ll_seg가 존재하는 조건 아래에 두어야 함

        for (x, y) in centerline:
            cv2.circle(vis_img, (int(x), int(y)), 2, (0, 255, 255), -1)



         # 제안 코드
        if handle_initialized:
            draw_handle_virtual_line(vis_img, 0)  # 0도 직선


        
        
        #if running:
            #apply_control(steer)
             # === 현재 핸들 상태 업데이트 ===
            #handle_angle += math.degrees(steer) * 0.1  # 누적 각도 갱신
            #현재 구조에서는 실제 핸들의 상태를 센서로 읽지 않으므로,steer 명령이 적용되면 핸들이 그만큼 돌아간 걸로 '가정'해야 함
            #math.degrees(steer)는 rad → degree 변환이고 ,0.1은 시간당 변화량 또는 적용율 조절을 위한 스케일러
            #너무 빨리 누적되면 0.05~0.1 사이로 줄이셈
        

        angle_to_lane = calculate_angle_from_centerline(centerline)   # B
        angle_error = angle_to_lane - handle_angle                    # B - A
        steer = pid_control(angle_error)                              # 조향 보정량

        if running:
            apply_control(steer)
            handle_angle += steer * 5.0  # 핸들 상태 업데이트 (steer는 라디안 단위)
            print(f"[DEBUG] 조향 입력: {math.degrees(steer):.2f}도, 누적 핸들: {math.degrees(handle_angle):.2f}도")
            handle_angle = max(min(handle_angle, math.radians(90)), math.radians(-90))  # 안전 제한


        cv2.imshow("YOLOPv2 Result", vis_img)
        cv2.imshow("Lane Mask", ll_seg * 255)
        cv2.moveWindow("YOLOPv2 Result", 1280, 100)
        cv2.moveWindow("Lane Mask", 1280, 600)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
"""
