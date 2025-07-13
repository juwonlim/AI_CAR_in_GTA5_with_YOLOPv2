# os 모듈: 파일 경로 및 디렉토리 관련 작업용
import os
# sys 모듈: 시스템 경로 및 명령줄 인자 처리
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

# grab_screen: 화면 캡처 기능 제공 (YOLOPv2 입력용)
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


# ========================== 조향 보조용 함수 ===========================
# 조향값 계산 함수 (기울기 기반)
# 중심선의 양 끝점을 이용해 조향 각도 계산 (arctan)
def calculate_angle_from_centerline(centerline_points):
    if len(centerline_points) < 2:
        return 0
    x1, y1 = centerline_points[0]
    x2, y2 = centerline_points[-1]
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return 0
    angle = np.arctan2(dy, dx)
    return angle

# 차선 마스크 기반으로 중심선 추출 + 시각화 포함
# YOLOPv2 차선 마스크로부터 중심선을 추출하고 시각화
def extract_centerline_from_mask(mask, output_img):
    h, w = mask.shape
    num_rows_to_scan = 50
    scan_rows = np.linspace(int(h * 0.5), h - 1, num_rows_to_scan).astype(int)

    left_points, right_points = [], []
    for row in scan_rows:
        indices = np.where(mask[row] > 0)[0]
        if len(indices) == 0:
            continue
        left = indices[indices < w // 2]
        right = indices[indices >= w // 2]
        if len(left) > 0:
            left_x = int(np.mean(left))
            left_points.append((left_x, row))
        if len(right) > 0:
            right_x = int(np.mean(right))
            right_points.append((right_x, row))

    center_points = []
    for i in range(len(scan_rows)):
        y = scan_rows[i]
        lx = np.polyval(np.polyfit([y for x, y in left_points], [x for x, y in left_points], 1), y) if len(left_points) >= 2 else None
        rx = np.polyval(np.polyfit([y for x, y in right_points], [x for x, y in right_points], 1), y) if len(right_points) >= 2 else None

        if lx is not None and rx is not None:
            cx = int((lx + rx) / 2)
        elif lx is not None:
            cx = int(lx + 80)
        elif rx is not None:
            cx = int(rx - 80)
        else:
            continue
        center_points.append((cx, y))

    # 시각화
    if output_img is not None:
        for x, y in left_points:
            cv2.circle(output_img, (x, y), 3, (255, 0, 0), -1)  # 파랑
        for x, y in right_points:
            cv2.circle(output_img, (x, y), 3, (0, 0, 255), -1)  # 빨강
        for x, y in center_points:
            cv2.circle(output_img, (x, y), 3, (0, 255, 255), -1)  # 노랑

    return center_points






# PID 제어로 조향 결정

# PID 제어에서 P 항만 고려한 간단한 조향 결정 함수
def pid_control(angle, kp=1.0):
    steering = kp * angle
    steering = max(min(steering, 1.0), -1.0)
    return steering


# 이미지 입력에 대해 도로/차선 세그멘테이션 마스크를 추출
def get_segmentation_masks(model, img, device):
    model.eval()
    img_resized = cv2.resize(img, (640, 360))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    da_seg = output['drivable'][0].argmax(0).byte().cpu().numpy()
    ll_seg = output['lane'][0].argmax(0).byte().cpu().numpy()

    da_seg = cv2.resize(da_seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    ll_seg = cv2.resize(ll_seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    return da_seg, ll_seg





# 메인 루프

# 웹캠 입력을 사용한 테스트용 메인 루프
def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        visual_img = frame.copy()
        da_seg_mask, ll_seg_mask = get_segmentation_masks(model, frame, device)

        centerline_points = extract_centerline_from_mask(ll_seg_mask, visual_img)
        angle = calculate_angle_from_centerline(centerline_points)
        steer_cmd = pid_control(angle)

        # 조향 적용
        if steer_cmd > 0.2:
            pyautogui.keyDown('d')
            pyautogui.keyUp('a')
        elif steer_cmd < -0.2:
            pyautogui.keyDown('a')
            pyautogui.keyUp('d')
        else:
            pyautogui.keyUp('a')
            pyautogui.keyUp('d')

        pyautogui.keyDown('w')

        output_img = show_seg_result(visual_img, (da_seg_mask, ll_seg_mask), is_demo=True)
        cv2.imshow("YOLOPv2 Result", output_img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


