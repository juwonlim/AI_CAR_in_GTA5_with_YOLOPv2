import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import time
import numpy as np
from keyboard_input_only_rev00 import controller
import win32ui
import win32con
import win32gui


import pydirectinput
import pydirectinput as pyautogui
pydirectinput.FAILSAFE = False
import torch


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
IMGSZ = 640


# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load(WEIGHTS_PATH).to(device)
model.half()
model.eval()

# GPU warm-up
model(torch.zeros(1, 3, IMGSZ, IMGSZ).to(device).half())

frame_count = 0

def get_current_speed_from_screen():
    screen = grab_screen()
    region = grab_speed_region(screen)  #수정: screen 전달
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, config='--psm 7 digits')
    try:
        speed = int(''.join(filter(str.isdigit, text)))
        return speed
    except:
        return None
    



def check_front_vehicle_distance(pred, image_shape, safe_distance_px=120):
    """
    pred: YOLO의 객체 탐지 결과
    image_shape: (h, w, c) → 중심 계산용
    safe_distance_px: 화면 기준으로 안전 거리 (픽셀 기준)
    """
    h, w, _ = image_shape
    mid_x = w // 2
    min_distance = None

    for det in pred:
        x1, y1, x2, y2, conf, cls = det
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if abs(cx - mid_x) < 80:  # 중심선 근처에 위치한 차량
            distance = y2 - y1
            if min_distance is None or distance > min_distance:
                min_distance = distance

    if min_distance and min_distance > safe_distance_px:
        return False  # 거리 충분
    return True  # 가까움 → 감속 필요





# 개선된 offset 계산 함수 (양쪽 차선 고려, 한쪽만 있을 때도 대응)
def calculate_offset_from_lane_mask(mask, fallback_offset):
    h, w = mask.shape
    midline = w // 2
    row = h // 2

    lane_indices = np.where(mask[row] == 1)[0]
    if len(lane_indices) == 0:
        return fallback_offset  # 차선이 없으면 이전 offset 유지

    # 왼쪽/오른쪽 차선 분리
    left = lane_indices[lane_indices < midline]
    right = lane_indices[lane_indices >= midline]

    if len(left) > 0 and len(right) > 0:
        # 양쪽 차선이 있는 경우 → 중앙선
        left_x = np.min(left)
        right_x = np.max(right)
        lane_center = (left_x + right_x) // 2
    else:
        # 한쪽 차선만 있는 경우 → 그쪽 차선에서 일정 거리 띄움
        if len(left) > 0:
            lane_center = np.min(left) + 150  # 오른쪽으로 일정 거리
        elif len(right) > 0:
            lane_center = np.max(right) - 150  # 왼쪽으로 일정 거리
        else:
            return fallback_offset

    return lane_center - midline




#동일함
def apply_control(offset, threshold=10, slow_down_zone=20, max_speed_kmh=60,slow_down=False):
    global frame_count
    frame_count += 1

    if offset is None:
        pyautogui.keyUp('w')
        pyautogui.keyUp('a')
        pyautogui.keyUp('d')
        return

    offset = np.clip(offset, -100, 100)
    current_speed = get_current_speed_from_screen()
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
    if current_speed is not None:
        if current_speed < max_speed_kmh - 5:
            pyautogui.keyDown('w')
        elif current_speed >= max_speed_kmh:
            pyautogui.keyUp('w')



    else:
        pyautogui.keyUp('w')

    if offset < -threshold:
        pyautogui.keyDown('a')
        pyautogui.keyUp('d')
        time.sleep(0.05)
    elif offset > threshold:
        pyautogui.keyDown('d')
        pyautogui.keyUp('a')
        time.sleep(0.05)
    else:
        pyautogui.keyUp('a')
        pyautogui.keyUp('d')
         
    if slow_down:
        pyautogui.keyUp('w')
        print("[INFO] 앞차와 거리 좁음 → 감속")
        return



#동일함
def stop_control():
    pyautogui.keyUp('w')
    pyautogui.keyUp('a')
    pyautogui.keyUp('d')






def main_drive_loop():
    
    global last_valid_offset #함수 안에서 전역변수에 값을 할당할 때	--> 반드시 global 선언 필요 , 읽기만 할 경우는 필요없음

    last_valid_offset = 0  # 차선이 사라졌을 때 쓸 기본값, NameError: name 'last_valid_offset' is not defined --> 이거 없으면 이 선언오류 발생
    #calculate_offset_from_lane_mask() 함수는 → 차선이 안 보일 때 fallback_offset을 사용
    #따라서 last_valid_offset이라는 기억 저장소가 필요
    #그런데 그것이 선언되지 않은 상태면 에러가 발생

    print("[INFO] 자율주행 시작 (Y/N 전환, ESC 종료)")

    while True:
        controller.check_key_events()

        if controller.is_exit_pressed():
            print("[INFO] ESC 입력 - 자율주행 종료")
            stop_control()
            break

        if controller.is_auto_drive_enabled():
            screen = grab_screen()
            visual_img = cv2.resize(screen, (1280, 720))
            model_input = letterbox(visual_img, IMGSZ, stride=32)[0]
            model_input = model_input[:, :, ::-1].transpose(2, 0, 1)
            model_input = np.ascontiguousarray(model_input)

            img_tensor = torch.from_numpy(model_input).to(device).half()
            img_tensor /= 255.0
            if img_tensor.ndimension() == 3:
                img_tensor = img_tensor.unsqueeze(0)

            with torch.no_grad():
                [pred, anchor_grid], seg, ll = model(img_tensor)

            #step 1: 세그멘테이션 결과를 visual_img에 오버레이
            # 도로/차선 색상 마스크 적용
            da_seg_mask = driving_area_mask(seg)
            ll_seg_mask = lane_line_mask(ll)

            #결과창 띄우기
            #output_img = show_seg_result(visual_img.copy(), (da_seg_mask, ll_seg_mask), is_demo=True)
            #cv2.namedWindow("YOLOPv2 Result", cv2.WINDOW_NORMAL)
            #cv2.resizeWindow("YOLOPv2 Result", 640, 360)
            #cv2.moveWindow("YOLOPv2 Result", 1280, 150)
            #cv2.imshow("YOLOPv2 Result", output_img)

            # 도로/차선 세그먼트 마스크를 직접 overlay --시작
            color_area = np.zeros_like(visual_img)
            color_area[da_seg_mask == 1] = [0, 255, 0]     # 도로 → 초록
            color_area[ll_seg_mask == 1] = [255, 0, 0]     # 차선 → 파랑
            color_mask = np.mean(color_area, 2)
            visual_img[color_mask != 0] = visual_img[color_mask != 0] * 0.5 + color_area[color_mask != 0] * 0.5

            """ 
            clip_coords(pred, visual_img.shape)
            for det in pred:
                *xyxy, conf, cls = det
                plot_one_box(xyxy, visual_img, label=f"Car {conf:.2f}")

            """

            offset = calculate_offset_from_lane_mask(ll_seg_mask, fallback_offset=last_valid_offset)
            output_img = draw_virtual_centerline(visual_img.copy(), offset)
            
            last_valid_offset = offset if offset is not None else last_valid_offset

            cv2.namedWindow("YOLOPv2 Result", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLOPv2 Result", 640, 360)
            cv2.moveWindow("YOLOPv2 Result", 1280, 150)
            cv2.imshow("YOLOPv2 Result", output_img)

            #객체 bbox, 중심선 모두 visual_img에 직접 표시
             #도로/차선 세그먼트 마스크를 직접 overlay --끝  


         

          

            # 2. 객체 검출 결과 후처리
            pred = split_for_trace_model(pred, anchor_grid)
            pred = non_max_suppression(pred)[0]  # <- 여기가 bbox만 추려낸 부분
            # bbox 클리핑 및 표시 (이 전에 꼭  후처리 먼저 되야 함)
            clip_coords(pred, visual_img.shape)

            for det in pred:
                *xyxy, conf, cls = det
                plot_one_box(xyxy, visual_img, label=f"Car {conf:.2f}")

      

          


            

            # step3 :차량 거리 확인
            slow_down_due_to_vehicle = check_front_vehicle_distance(pred, visual_img.shape)

            apply_control(offset, slow_down=slow_down_due_to_vehicle)
            #여기까지 차간거리 유지 통합

            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            stop_control()
            time.sleep(0.05)


if __name__ == '__main__':
    main_drive_loop()






###rev02 이전의 함수

""" 
#yolopv2에 신규추가 (기존 calculate_steering_from_fit 대체??)

def calculate_offset_from_lane_mask(mask):
    h, w = mask.shape                        # 마스크의 높이, 너비 얻기
    midline = w // 2                         # 화면의 수평 중앙선 계산
    lane_indices = np.where(mask[h//2] == 1)[0]  # 화면 수직 절반 높이에서 차선 픽셀들의 x좌표 추출
    if len(lane_indices) == 0:              # 차선이 검출되지 않았을 경우
        return None                         # None 리턴하여 제어 중지 유도
    lane_center = int(np.mean(lane_indices)) # 검출된 차선 x좌표들의 평균값 → 중심 추정
    return lane_center - midline            # 중심점과 실제 중앙의 차이값 (offset)

"""


""" 
def main_drive_loop():
    print("[INFO] 자율주행 시작 (Y/N 전환, ESC 종료)")

    while True:
        controller.check_key_events()  # 키 입력 체크 (Y/N/ESC)

        if controller.is_exit_pressed():
            print("[INFO] ESC 입력 - 자율주행 종료")
            stop_control()
            break

        if controller.is_auto_drive_enabled():
            screen = grab_screen()  # 화면 캡처
            visual_img = cv2.resize(screen, (1280, 720))  # 모델 입력 사이즈 맞춤
            model_input = letterbox(visual_img, IMGSZ, stride=32)[0]  # YOLO에 맞는 정사각 크기 + padding
            model_input = model_input[:, :, ::-1].transpose(2, 0, 1)  # RGB→BGR, 그리고 channel first로 변경
            model_input = np.ascontiguousarray(model_input)

            img_tensor = torch.from_numpy(model_input).to(device).half()  # 텐서 변환 후 GPU로 이동
            img_tensor /= 255.0
            if img_tensor.ndimension() == 3:
                img_tensor = img_tensor.unsqueeze(0)  # 배치 차원 추가

            with torch.no_grad():
                [pred, anchor_grid], seg, ll = model(img_tensor)  # YOLOPv2 추론 실행

            da_seg_mask = driving_area_mask(seg)      # 주행 가능 영역 추출
            ll_seg_mask = lane_line_mask(ll)          # 차선 마스크 추출

            offset = calculate_offset_from_lane_mask(ll_seg_mask)  # 중심 오차 계산
            apply_control(offset)                    # 오차 기반 조향 제어 수행

            output_img = draw_virtual_centerline(visual_img.copy(), offset)  # 시각화 라인
            cv2.imshow("YOLOPv2 Drive View", output_img)  # 화면 출력

            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            stop_control()  # 자율주행 OFF 시 모든 키 해제
            time.sleep(0.05)
"""

