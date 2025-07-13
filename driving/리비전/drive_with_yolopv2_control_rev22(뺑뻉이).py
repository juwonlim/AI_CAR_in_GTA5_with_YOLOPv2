

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import time
import numpy as np
from keyboard_input_only_rev00 import controller #  키보드 이벤트 제어
import win32ui
import win32con
import win32gui


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
    



# PID 제어용 변수
last_error = 0
integral = 0


# 차선 마스크 기반으로 중심선 추출 + 시각화 포함
def extract_centerline_from_mask(mask, output_img):
     # 입력된 마스크 이미지의 높이(h), 너비(w) 추출
    h, w = mask.shape
    num_rows_to_scan = 50 ## 하단 절반 영역에서 50개의 스캔 라인(y좌표)을 균등하게 생성
    scan_rows = np.linspace(int(h * 0.5), h - 1, num_rows_to_scan).astype(int)

    left_points, right_points = [], [] # # 좌/우 차선 점들을 저장할 리스트 초기화
    
       # 각 스캔 라인(row)에서 차선 픽셀을 탐색
    for row in scan_rows:
          # 해당 row에서 마스크 값이 0보다 큰 (즉, 차선으로 인식된) 인덱스 추출
        indices = np.where(mask[row] > 0)[0]
        if len(indices) == 0:
            continue  # 아무 차선 픽셀이 없으면 skip

        # 좌측과 우측 차선을 나누기 위한 기준은 화면의 중앙 (w // 2)
        left = indices[indices < w // 2]
        right = indices[indices >= w // 2]
        
         # 좌측 차선이 있으면 평균 x좌표를 계산하여 포인트로 저장
        if len(left) > 0:
            left_x = int(np.mean(left))
            left_points.append((left_x, row))
        
        # 우측 차선이 있으면 평균 x좌표를 계산하여 포인트로 저장
        if len(right) > 0:
            right_x = int(np.mean(right))
            right_points.append((right_x, row))

    
     # 중심선을 구성할 포인트 리스트
    center_points = []

     # scan_rows와 동일한 y좌표 순서로 중심선 포인트 계산
    for i in range(len(scan_rows)):
        y = scan_rows[i]

          # 좌/우 차선에 대해 직선 근사 (x = a*y + b 형태로 근사하여 x 좌표 추정) --> 탄젠트하게 붙는 직선을 만드는 코드
        lx = np.polyval(np.polyfit([y for x, y in left_points], [x for x, y in left_points], 1), y) if len(left_points) >= 2 else None
        rx = np.polyval(np.polyfit([y for x, y in right_points], [x for x, y in right_points], 1), y) if len(right_points) >= 2 else None

        
        # 좌우 차선이 모두 존재하면 두 선의 중간을 중심선으로 사용
        if lx is not None and rx is not None:
            cx = int((lx + rx) / 2)

        # 좌측 차선만 있을 경우 오른쪽으로 80픽셀 이동한 위치를 중심선으로 가정   
        elif lx is not None:
            cx = int(lx + 80)

         # 우측 차선만 있을 경우 왼쪽으로 80픽셀 이동한 위치를 중심선으로 가정    
        elif rx is not None:
            cx = int(rx - 80)
        else:
            continue # 둘 다 없으면 중심선 포인트 계산 불가
        center_points.append((cx, y))

        # 시각화 처리
    if output_img is not None:

         # 좌측 차선 포인트: 파란색
        for x, y in left_points:
            cv2.circle(output_img, (x, y), 3, (255, 0, 0), -1)  # 파랑
        
        # 우측 차선 포인트: 빨간색
        for x, y in right_points:
            cv2.circle(output_img, (x, y), 3, (0, 0, 255), -1)  # 빨강
        
        # 중심선 포인트: 노란색
        for x, y in center_points:
            cv2.circle(output_img, (x, y), 3, (0, 255, 255), -1)  # 노랑 --> extract_centerline_from_mask() 함수 내에서 중심선 포인트가 시각화 됨
                                                                  #노란색 점들이 중심선을 형성하며, 이를 통해 어떤 선을 따라가는지 눈으로 확인 가능

    
    return center_points # 최종적으로 중심선 포인트 리스트 반환 (centerline_points는 위에서 계산된 lx, rx를 통해 생성된 가상의 중심점들의 리스트)


# ========================== 조향 보조용 함수 ===========================
# 조향값 계산 함수 (기울기 기반)
def calculate_angle_from_centerline(centerline_points):
     # 중심선 포인트가 2개 미만이면 기울기를 계산할 수 없으므로 0을 반환
    if len(centerline_points) < 2:
        return 0
    
     # 첫 번째 점 (화면 하단)과 마지막 점 (화면 상단)을 가져옴
    x1, y1 = centerline_points[0] #하단 점
    x2, y2 = centerline_points[-1] #상단 점
    #그 기울기를 arctangent로 변환(라디안) 하여 반환
    
     # 두 점 간의 x축과 y축 거리 계산
    dx = x2 - x1
    dy = y2 - y1

     # dx가 0이면 수직이므로 기울기를 계산할 수 없으므로 0 반환 (오버슈팅 방지)
    if dx == 0:
        return 0
    
    # 중심선의 기울기(방향)를 arctangent로 계산하여 라디안 단위로 반환
    # atan2는 dx가 0일 때도 안정적으로 작동하므로 위의 if는 안전장치 성격

    angle = np.arctan2(dy, dx)  #extract_centerline_from_mask함수의 리턴값center_points를 받아서 그 포인트의 첫점과 마지막 점의 방향을 계산해 angle반환
    return angle  # 라디안 단위의 방향각을 반환 , 이 angle은 pid_control(angle)로 전달되어 steering 값 계산 → 최종적으로 조향 결정에 사용











# PID 제어로 조향 결정
"""
이 함수는 단순한 비례제어(P control) 방식입

중심선의 기울기(angle) 값을 받아서 조향 명령을 생성

기울기가 크면 차량이 많이 틀어야 하므로 큰 조향값을 주고, 작으면 작은 조향값을 줌

조향값의 범위를 -1.0(최대 좌회전)에서 +1.0(최대 우회전)으로 제한하여 제어 안정성을 확보

"""

def pid_control(angle, kp=1.0):
    # 입력된 각도(angle)에 비례 계수(kp)를 곱해 조향값 계산
    # 이때 angle은 중심선의 기울기 (radian)이며, 양수는 우회전, 음수는 좌회전을 의미
    steering = kp * angle
    
    
    # 조향값을 [-1.0, 1.0] 범위로 제한
    # 너무 급격한 조향을 방지하고, 차량 조작을 안정적으로 유지하기 위해 사용
    steering = max(min(steering, 1.0), -1.0)
    return steering    # 최종 조향값 반환
 


def get_segmentation_masks(model, img, device):
    model.eval()

    img_resized = letterbox(img, new_shape=640)[0]
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device).half()

    with torch.no_grad():
        output = model(img_tensor)

    # Step 1: Unpack output
    if isinstance(output, dict):
        drivable_output = output['drivable']
        lane_output = output['lane']
    elif isinstance(output, (tuple, list)):
        drivable_output, lane_output = output[:2]
    else:
        raise TypeError(f"[ERROR] Unexpected output type: {type(output)}")

    # Step 2: Unwrap if list or tuple
    for name, out in [('drivable', drivable_output), ('lane', lane_output)]:
        if isinstance(out, (list, tuple)):
            out = out[0]
        if isinstance(out, (list, tuple)):
            out = out[0]
        if name == 'drivable':
            drivable_output = out
        else:
            lane_output = out

    # Step 3: Remove batch dim if needed
    if drivable_output.ndim == 4:
        drivable_output = drivable_output[0]
    if lane_output.ndim == 4:
        lane_output = lane_output[0]

    # Step 4: Argmax → segmentation mask
    da_seg = drivable_output.argmax(0).byte().cpu().numpy()
    ll_seg = lane_output.argmax(0).byte().cpu().numpy()

    # Step 5: Resize to original image size
    da_seg = cv2.resize(da_seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    ll_seg = cv2.resize(ll_seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    return da_seg, ll_seg



def stop_control():
    pyautogui.keyUp('w')
    pyautogui.keyUp('a')
    pyautogui.keyUp('d')


def main():

  
  

    while True:
        print("[INFO] 자율주행 시작 (Y/N 전환, ESC 종료)")
        controller.check_key_events()
        if controller.is_exit_pressed():
            stop_control()
            break
        if controller.is_auto_drive_enabled():
            # 자율주행 로직 실행
        else:
            stop_control()
            time.sleep(0.05)

        
        frame = grab_screen()
        frame_resized = cv2.resize(frame, (1280, 720))  # 화면 크기 조정
        input_img = letterbox(frame_resized, new_shape=640)[0]

        visual_img = input_img.copy() # 시각화용 이미지 복사
        da_seg_mask, ll_seg_mask = get_segmentation_masks(model, frame, device)  # YOLOPv2 모델을 통해 주행 가능 영역(da)과 차선 영역(ll) 마스크 추출

        centerline_points = extract_centerline_from_mask(ll_seg_mask, visual_img)  # 차선 마스크를 기반으로 중심선 포인트 추출 (좌/우 차선 → 중앙 선산출)
        angle = calculate_angle_from_centerline(centerline_points) # 중심선 포인트를 기반으로 조향 각도 계산
        steer_cmd = pid_control(angle)  # PID 제어기로 조향 명령 계산 (-1 ~ 1 사이 값)

        # ─── 조향 명령을 실제 키 입력으로 변환 ───
        if steer_cmd > 0.2:
            pyautogui.keyDown('d')  # 오른쪽 회전
            pyautogui.keyUp('a')  # 왼쪽 키 해제
        elif steer_cmd < -0.2:
            pyautogui.keyDown('a') # 왼쪽 회전
            pyautogui.keyUp('d')   # 오른쪽 키 해제
        else:
            pyautogui.keyUp('a')  # 중립 (직진)
            pyautogui.keyUp('d')

        pyautogui.keyDown('w')    # 전진 가속 유지


         # YOLOPv2 결과 시각화 (세그멘테이션 마스크 오버레이)
        output_img = show_seg_result(visual_img, (da_seg_mask, ll_seg_mask), is_demo=True)
        
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("result", 640, 360)
        cv2.imshow("result", output_img)
        cv2.moveWindow("result", 1280, 150)



          # ESC 키 입력 시 루프 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break
   
   
    # 자원 해제        
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()





