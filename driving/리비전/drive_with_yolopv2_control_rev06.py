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


# 전역 변수로 마지막 속도 저장
last_known_speed = 0

def get_current_speed_from_screen():
 
    screen = grab_screen()
    region = grab_speed_region(screen)  #수정: screen 전달
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) # 흑백 변환
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)  #명암 이진화
    text = pytesseract.image_to_string(thresh, config='--psm 7 digits') # OCR로 숫자 추출

    try:
        speed = int(''.join(filter(str.isdigit, text)))  # 숫자만 골라서 정수로 변환
        global last_known_speed
        last_known_speed = speed  # # 성공한 속도는 저장해둠
        return speed
    except:
        return last_known_speed   # OCR 실패하면 마지막 속도를 리턴
    



def check_front_vehicle_distance(pred, image_shape, safe_distance_px=120):
    """

    pred: YOLO의 객체 탐지 결과
    image_shape: (h, w, c) → 중심 계산용
    safe_distance_px: 화면 기준으로 안전 거리 (픽셀 기준)
    """
    h, w, _ = image_shape
    mid_x = w // 2 # 화면 중앙 x좌표
    min_distance = None  # 가장 가까운 물체의 높이 기록용

    for det in pred:
        x1, y1, x2, y2, conf, cls = det
        cx = int((x1 + x2) / 2)  # 중심 x좌표
        cy = int((y1 + y2) / 2)   # 중심 y좌표

        # 개선 필터
        # 다음 조건을 만족하는 bbox만 판단
        if int(cls) != 2: # 자동차 클래스가 아니면 제외
            continue
        if conf < 0.6: # 신뢰도가 낮으면 제외
            continue
        if abs(cx - mid_x) > 60: # 중심선에서 너무 멀면 제외
            continue
        if (y2 - y1) < 30:  # 너무 작으면 제외
            continue

        #  여기서 거리 계산
        distance = y2 - y1  # bbox의 높이 → 가까운 차일수록 높이가 큼

        if min_distance is None or distance > min_distance:
            min_distance = distance  # 가장 큰(가까운) bbox 저장

    if min_distance and min_distance > safe_distance_px:
        return False   # 충분히 멀면 감속 안 함

    return True  # 가까우면 감속해야 함





# 개선된 offset 계산 함수 (양쪽 차선 고려, 한쪽만 있을 때도 대응)
def calculate_offset_from_lane_mask(mask, fallback_offset):
    h, w = mask.shape
    midline = w // 2
    row = h // 2

    lane_indices = np.where(mask[row] == 1)[0] # 중간 행의 차선 픽셀 인덱스
    if len(lane_indices) == 0:
        return fallback_offset  # 차선이 없으면 이전 offset 유지

    # 왼쪽/오른쪽 차선 분리
    left = lane_indices[lane_indices < midline]  # 왼쪽 차선
    right = lane_indices[lane_indices >= midline]  # 오른쪽 차선
 
    if len(left) > 0 and len(right) > 0:
         # 양쪽 차선이 있으면 → 가운데 계산
        left_x = np.min(left)
        right_x = np.max(right)
        lane_center = (left_x + right_x) // 2
    else:
        # 한쪽 차선만 있는 경우 → 그 쪽 차선에서 일정 거리 띄움
        if len(left) > 0:
            lane_center = np.min(left) + 80  # 오른쪽으로 일정 거리 ,이전 값 150
        elif len(right) > 0:
            lane_center = np.max(right) - 80  # 왼쪽으로 일정 거리 , 이전값 150
        else:
            return fallback_offset

    return lane_center - midline  # 화면 중심 기준 offset




#동일함
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
        controller.check_key_events() # 키보드 이벤트 감지 (Y/N/ESC)

        if controller.is_exit_pressed(): # ESC 눌렸다면
            print("[INFO] ESC 입력 - 자율주행 종료") 
            stop_control() # 모든 키 release
            break

        if controller.is_auto_drive_enabled():  # Y 키 눌러 자율주행 활성화된 경우
            screen = grab_screen()
            visual_img = cv2.resize(screen, (1280, 720))

            # YOLOPv2 입력 전처리
            model_input = letterbox(visual_img, IMGSZ, stride=32)[0]
            model_input = model_input[:, :, ::-1].transpose(2, 0, 1) # RGB → BGR, 채널 순서 변경
            model_input = np.ascontiguousarray(model_input)

             # 텐서 변환 및 모델 입력
            img_tensor = torch.from_numpy(model_input).to(device).half()
            img_tensor /= 255.0 # 0~1 스케일링
            if img_tensor.ndimension() == 3:
                img_tensor = img_tensor.unsqueeze(0)  # 배치 차원 추가

            with torch.no_grad(): # 추론 시 gradient 저장 안 함
                [pred, anchor_grid], seg, ll = model(img_tensor)  # YOLOPv2 모델 추론

            #step 1: 세그멘테이션 결과를 visual_img에 오버레이
            # 도로/차선 색상 마스크 적용
            da_seg_mask = driving_area_mask(seg)  # 도로 영역 마스크
            ll_seg_mask = lane_line_mask(ll)  # 차선 영역 마스크

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
             # 차선 offset 계산 (없으면 이전 값 사용)
            offset = calculate_offset_from_lane_mask(ll_seg_mask, fallback_offset=last_valid_offset)
            output_img = draw_virtual_centerline(visual_img.copy(), offset)  # 중앙선 그리기

            # 차량 거리 확인
            slow_down_due_to_vehicle = check_front_vehicle_distance(pred, visual_img.shape)  # 앞차 있음?
            # 실제 조향 및 속도 적용 → current_speed 리턴됨
            current_speed = apply_control(offset, slow_down=slow_down_due_to_vehicle) # 제어 수행

            last_valid_offset = offset if offset is not None else last_valid_offset  # 유효 offset 저장

            #output_img 위에 offset, 속도, 조향 상태 등을 실시간 출력
            cv2.putText(output_img, f"Offset: {offset}", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(output_img, f"Speed: {current_speed}", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            #결과창 띄우기
            cv2.namedWindow("YOLOPv2 Result", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLOPv2 Result", 640, 360)
            cv2.moveWindow("YOLOPv2 Result", 1280, 150)
            cv2.imshow("YOLOPv2 Result", output_img)

            #객체 bbox, 중심선 모두 visual_img에 직접 표시
             #도로/차선 세그먼트 마스크를 직접 overlay --끝  


         

          

           # 객체감지 처리
            pred = split_for_trace_model(pred, anchor_grid) # 앵커 기반 bbox 복원
            pred = non_max_suppression(pred)[0]  # NMS로 겹치는 bbox 제거
            # bbox 클리핑 및 표시 (이 전에 꼭  후처리 먼저 되야 함)
            clip_coords(pred, visual_img.shape) # 화면 범위 벗어난 bbox 좌표 조정

            for det in pred:
                *xyxy, conf, cls = det
                plot_one_box(xyxy, visual_img, label=f"Car {conf:.2f}")  # bbox 그리기

      

          


            

            

           

            #여기까지 차간거리 유지 통합

            if cv2.waitKey(1) & 0xFF == 27:
                break # ESC 눌리면 종료
        else:
            stop_control() # 자동운전 비활성화 시 키 입력 모두 해제
            time.sleep(0.05) # 50ms 대기


if __name__ == '__main__':
    main_drive_loop()



