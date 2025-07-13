

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

# 자율주행 키 입력 감지 함수 (GTA 창에서 작동)
def check_autonomous_key():
    global autonomous_mode
    # Y 키: 자율주행 ON
    if win32api.GetAsyncKeyState(ord('Y')) & 0x8000:
        autonomous_mode = True
    # N 키: 자율주행 OFF
    if win32api.GetAsyncKeyState(ord('N')) & 0x8000:
        autonomous_mode = False
    # ESC 키: 종료
    if win32api.GetAsyncKeyState(27) & 0x8000:
        return True
    return False

# 조향 제어 함수
def apply_control(steering, speed=1.0):
    pyautogui.keyDown('w')  # 전진
    if steering > 0.2:
        pyautogui.keyDown('d')
        pyautogui.keyUp('a')
    elif steering < -0.2:
        pyautogui.keyDown('a')
        pyautogui.keyUp('d')
    else:
        pyautogui.keyUp('a')
        pyautogui.keyUp('d')

# 중심선 각도 계산 함수
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

# 중심선 추출 함수 (왼쪽/오른쪽 차선 기반)
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

    if output_img is not None:
        for x, y in left_points:
            cv2.circle(output_img, (x, y), 3, (255, 0, 0), -1)
        for x, y in right_points:
            cv2.circle(output_img, (x, y), 3, (0, 0, 255), -1)
        for x, y in center_points:
            cv2.circle(output_img, (x, y), 3, (0, 255, 255), -1)

    return center_points

# PID 제어 함수 (단순 비례 제어)
def pid_control(angle, kp=1.0):
    steering = kp * angle
    return max(min(steering, 1.0), -1.0)

# YOLOPv2 마스크 추출 함수
def get_segmentation_masks(model, img, device):
    model.eval()
    img_resized = letterbox(img, new_shape=640)[0]
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device).half()

    with torch.no_grad():
        output = model(img_tensor)

    if isinstance(output, dict):
        da_seg = output['drivable'][0].argmax(0).byte().cpu().numpy()
        ll_seg = output['lane'][0].argmax(0).byte().cpu().numpy()
    elif isinstance(output, tuple):
        drivable_output, lane_output = output[:2]
        if isinstance(drivable_output, list):
            drivable_output = drivable_output[0]
        if isinstance(lane_output, list):
            lane_output = lane_output[0]
        da_seg = drivable_output.argmax(0).byte().cpu().numpy()
        ll_seg = lane_output.argmax(0).byte().cpu().numpy()
    else:
        raise TypeError(f"[ERROR] 예상치 못한 출력 타입: {type(output)}")

    da_seg = cv2.resize(da_seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    ll_seg = cv2.resize(ll_seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    return da_seg, ll_seg

# 메인 루프
def main():
    global autonomous_mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    print("[INFO] 자율주행 대기 중... (Y/N 전환, ESC 종료)")

    while True:
        if check_autonomous_key():
            print("[INFO] ESC 입력됨. 종료합니다.")
            break

        frame = grab_screen(region=(0, 0, 1280, 720))
        frame = cv2.resize(frame, (640, 480))
        visual_img = frame.copy()

        if autonomous_mode:
            da_seg_mask, ll_seg_mask = get_segmentation_masks(model, frame, device)
            centerline_points = extract_centerline_from_mask(ll_seg_mask, visual_img)
            angle = calculate_angle_from_centerline(centerline_points)
            steer_cmd = pid_control(angle)
            apply_control(steer_cmd)

        output_img = show_seg_result(visual_img, (da_seg_mask, ll_seg_mask), is_demo=True)
        cv2.namedWindow("YOLOPv2 Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLOPv2 Result", 640, 360)
        cv2.moveWindow("YOLOPv2 Result", 1280, 150)
        cv2.imshow("YOLOPv2 Result", output_img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
