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
    





# ========================== 조향 보조용 함수 ===========================
#calculate_offset_from_lane_mask() 개선 (한쪽 차선만 있어도 가속)

def calculate_offset_from_lane_mask(mask, fallback_offset=0):
    # 입력된 차선 마스크의 높이(h)와 너비(w)를 가져옴
    h, w = mask.shape

    # 하단 절반~맨 아래까지 20개의 스캔 라인을 선택
    num_rows_to_scan = 20
    scan_rows = np.linspace(int(h * 0.5), h - 1, num_rows_to_scan).astype(int)

    # 왼쪽 차선과 오른쪽 차선의 x좌표들, 그리고 각 줄의 y좌표 저장 리스트 초기화
    left_xs, right_xs, ys = [], [], []

    for row in scan_rows:
        indices = np.where(mask[row] > 0)[0]  # 현재 줄(row)에서 차선 픽셀(x 위치) 찾기
        if len(indices) > 0:
            left = indices[indices < w // 2]   # 화면 왼쪽 절반에 있는 픽셀들
            right = indices[indices >= w // 2] # 화면 오른쪽 절반에 있는 픽셀들

            # 각 줄마다 평균 위치 저장 (없으면 None)
            left_xs.append(np.mean(left) if len(left) > 0 else None)
            right_xs.append(np.mean(right) if len(right) > 0 else None)
            ys.append(row)  # 해당 줄의 y좌표 저장

    # 중심 x좌표 리스트와 마지막으로 유효했던 중심 위치 저장용
    center_xs = []
    last_known = None

    # 각 줄마다 중심 x좌표 계산
    for l, r in zip(left_xs, right_xs):
        if l is not None and r is not None:
            # 양쪽 차선이 있으면 가운데를 중심으로
            c = (l + r) / 2
            last_known = c
        elif l is not None:
            # 왼쪽 차선만 있으면 오른쪽으로 80px offset → 가상의 중앙선
            c = l + 80
            last_known = c
        elif r is not None:
            # 오른쪽 차선만 있으면 왼쪽으로 80px offset → 가상의 중앙선
            c = r - 80
            last_known = c
        else:
            # 차선이 둘 다 없으면 이전 중심 값을 유지
            c = last_known
        center_xs.append(c)

    # None이 아닌 유효한 중심점들만 x, y 튜플로 저장
    valid_centers = [(int(x), int(y)) for x, y in zip(center_xs, ys) if x is not None]

    # 유효한 중심점이 너무 적으면 차선이 없다고 판단
    if len(valid_centers) < 5:
        print("[INFO] 차선 없음 → 정지")
        return fallback_offset, False

    # 하단의 마지막 중심점 기준으로 offset 계산
    center_x = valid_centers[-1][0]
    offset = center_x - w // 2  # 화면 중심 기준 → 양수면 오른쪽, 음수면 왼쪽

    # 차선 인식 상태에 따른 디버깅 메시지 출력
    if all([l is not None and r is not None for l, r in zip(left_xs, right_xs)]):
        print("[INFO] 양쪽 차선 인식 → 가속")
    elif any([l is not None for l in left_xs]):
        print("[INFO] 왼쪽 차선만 인식 → 가속")
    elif any([r is not None for r in right_xs]):
        print("[INFO] 오른쪽 차선만 인식 → 가속")
    else:
        print("[INFO] 차선 인식 실패")
        return fallback_offset, False

    # 최종 오프셋과 가속 조건 (True) 반환
    return offset, True





#draw_virtual_centerline() 개선 (곡선 중심선 시각화)
def draw_virtual_centerline(mask, output_img):
    h, w = mask.shape
    num_rows_to_scan = 20
    scan_rows = np.linspace(int(h * 0.5), h - 1, num_rows_to_scan).astype(int)

    left_xs, right_xs, ys = [], [], []
    center_points = []

    for row in scan_rows:
        indices = np.where(mask[row] > 0)[0]
        if len(indices) > 0:
            left = indices[indices < w // 2]
            right = indices[indices >= w // 2]

            l = np.mean(left) if len(left) > 0 else None
            r = np.mean(right) if len(right) > 0 else None

            if l is not None and r is not None:
                c = (l + r) / 2
            elif l is not None:
                c = l + 80
            elif r is not None:
                c = r - 80
            else:
                c = None

            if c is not None:
                center_points.append((int(c), int(row)))

    # 가상 중앙선 시각화
    if len(center_points) >= 2:
        cv2.polylines(output_img, [np.array(center_points)], isClosed=False, color=(0, 255, 255), thickness=2)

    return output_img, center_points





def apply_control(offset, trigger_drive):
    if trigger_drive:
        pyautogui.keyDown('w')
    else:
        pyautogui.keyUp('w')

    if offset < -30:
        pyautogui.keyDown('a')
        pyautogui.keyUp('d')
    elif offset > 30:
        pyautogui.keyDown('d')
        pyautogui.keyUp('a')
    else:
        pyautogui.keyUp('a')
        pyautogui.keyUp('d')




#동일함
def stop_control():
    pyautogui.keyUp('w')
    pyautogui.keyUp('a')
    pyautogui.keyUp('d')





# ======================== 주 제어 루프 =========================
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
            #offset = calculate_offset_from_lane_mask(ll_seg_mask, fallback_offset=last_valid_offset)

             # 1. 속도 측정
            current_speed = get_current_speed_from_screen()

            # 2. 차선 offset 계산
            offset, trigger_drive = calculate_offset_from_lane_mask(ll_seg_mask, fallback_offset=last_valid_offset)
           
            # 예시 1: 급격한 변화 완화
            if abs(offset - last_valid_offset) > 80:
                offset = (offset + last_valid_offset) / 2

            # 예시 2: 절대값 제한 , offset: 현재 프레임에서 계산된 조향용 오프셋
            offset = np.clip(offset, -100, 100)

            # 갱신, last_valid_offset: 이전 프레임에서 사용한 오프셋 (전역 변수로 선언 필요)
            last_valid_offset = offset

            # 3. 가속 및 조향 제어
            apply_control(offset, trigger_drive)

            # 4. 가상 중앙선 그리기
            output_img, _ = draw_virtual_centerline(ll_seg_mask, visual_img.copy())



            # 객체감지 처리
            pred = split_for_trace_model(pred, anchor_grid) # 앵커 기반 bbox 복원  --> yolo로 후처리된 결과를 pred에 저장 ,slow_down_due_to_vehicel나 current 밑에 있을 경우 'ValueError: not enough values to unpack (expected 6, got 1)' 이런 에러 발생
            pred = non_max_suppression(pred)[0]  # NMS로 겹치는 bbox 제거
            # bbox 클리핑 및 표시 (이 전에 꼭  후처리 먼저 되야 함)
            clip_coords(pred, visual_img.shape) # 화면 범위 벗어난 bbox 좌표 조정

            # bbox 디버그 표시
            for det in pred:
                *xyxy, conf, cls = det
                plot_one_box(xyxy, visual_img, label=f"Car {conf:.2f}")


            # 차량 거리 확인
            #slow_down_due_to_vehicle = check_front_vehicle_distance(pred, visual_img.shape)  # 앞차 있음?
            slow_down_due_to_vehicle = False  # 차량 감지 대신 항상 전진하도록 설정
            # 실제 조향 및 속도 적용 → current_speed 리턴됨
            #apply_control(offset, trigger_drive)
            #current_speed = apply_control(offset, slow_down=slow_down_due_to_vehicle) # 제어 수행

            last_valid_offset = offset if offset is not None else last_valid_offset  # 유효 offset 저장

            # 5. 디버깅 출력
            cv2.putText(output_img, f"Offset: {offset:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(output_img, f"Speed: {current_speed}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            #output_img 위에 offset, 속도, 조향 상태 등을 실시간 출력
            #cv2.putText(output_img, f"Offset: {offset}", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            #cv2.putText(output_img, f"Speed: {current_speed}", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            #결과창 띄우기
            cv2.namedWindow("YOLOPv2 Result", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLOPv2 Result", 640, 360)
            cv2.moveWindow("YOLOPv2 Result", 1280, 150)
            cv2.imshow("YOLOPv2 Result", output_img)

            #객체 bbox, 중심선 모두 visual_img에 직접 표시
             #도로/차선 세그먼트 마스크를 직접 overlay --끝  

        

            

           

            #여기까지 차간거리 유지 통합

            if cv2.waitKey(1) & 0xFF == 27:
                break # ESC 눌리면 종료
        else:
            stop_control() # 자동운전 비활성화 시 키 입력 모두 해제
            time.sleep(0.05) # 50ms 대기


if __name__ == '__main__':
    main_drive_loop()



