import cv2
import numpy as np
import torch
import time
import os
import pyautogui
from collections import deque
from preprocess_for_yolopv2 import grab_screen, letterbox
from models.experimental import attempt_load

# 적용된 키 목록 (조향 제어용)
key_list = ['w', 'a', 'd']
prev_steer = 0  # 이전 조향값 저장용

# 차량 속도 영역 정의 (우측 하단 사각형)
speed_region = (1165, 660, 100, 50)  # (x, y, w, h)

# ▶ H 키로 초기 핸들 직선 설정
handle_line_defined = False
handle_line_angle = 0
handle_line_pts = []

# ▶ Y 키로 자율주행 시작
autonomous_mode = False

def get_segmentation_masks(model, img, device):
    model.eval()
    img_resized = letterbox(img, new_shape=640)[0]
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device).half()

    with torch.no_grad():
        output = model(img_tensor)

    if isinstance(output, (list, tuple)):
        drivable_output = output[0]
        lane_output = output[1]
        if isinstance(drivable_output, list):
            drivable_output = drivable_output[0]
        if isinstance(lane_output, list):
            lane_output = lane_output[0]
        da_seg = drivable_output.argmax(0).byte().cpu().numpy()
        ll_seg = lane_output.argmax(0).byte().cpu().numpy()
    else:
        raise TypeError(f"[ERROR] 예기치 못한 출력 타입: {type(output)}")

    da_seg = cv2.resize(da_seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    ll_seg = cv2.resize(ll_seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return da_seg, ll_seg

def extract_centerline_from_mask(mask):
    height, width = mask.shape
    roi = mask[int(height * 0.6):, :]  # 하단 40%만 사용

    left_points, right_points = [], []
    for y in range(roi.shape[0]):
        row = roi[y]
        xs = np.where(row > 0)[0]
        if len(xs) >= 2:
            left_x = xs[0]
            right_x = xs[-1]
            left_points.append((left_x, y + int(height * 0.6)))
            right_points.append((right_x, y + int(height * 0.6)))

    center_points = []
    for (lx, y1), (rx, y2) in zip(left_points, right_points):
        cx = int((lx + rx) / 2)
        center_points.append((cx, y1))
    return center_points, left_points, right_points

def calculate_angle_from_centerline(centerline_points):
    if len(centerline_points) < 10:
        return 0.0

    head = np.mean(centerline_points[:5], axis=0)
    tail = np.mean(centerline_points[-5:], axis=0)
    dx = tail[0] - head[0]
    dy = tail[1] - head[1]
    angle = np.arctan2(dy, dx)
    return angle
############여기까지 1부##################

import cv2
import numpy as np
import torch
import time
import os
import pyautogui
from collections import deque
from preprocess_for_yolopv2 import grab_screen, letterbox
from models.experimental import attempt_load

# 적용된 키 목록 (조향 제어용)
key_list = ['w', 'a', 'd']
prev_steer = 0  # 이전 조향값 저장용

# 차량 속도 영역 정의 (우측 하단 사각형)
speed_region = (1165, 660, 100, 50)  # (x, y, w, h)

# ▶ H 키로 초기 핸들 직선 설정
handle_line_defined = False
handle_line_angle = 0
handle_line_pts = []

# ▶ Y 키로 자율주행 시작
autonomous_mode = False

def get_segmentation_masks(model, img, device):
    model.eval()
    img_resized = letterbox(img, new_shape=640)[0]
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device).half()

    with torch.no_grad():
        output = model(img_tensor)

    if isinstance(output, (list, tuple)):
        drivable_output = output[0]
        lane_output = output[1]
        if isinstance(drivable_output, list):
            drivable_output = drivable_output[0]
        if isinstance(lane_output, list):
            lane_output = lane_output[0]
        da_seg = drivable_output.argmax(0).byte().cpu().numpy()
        ll_seg = lane_output.argmax(0).byte().cpu().numpy()
    else:
        raise TypeError(f"[ERROR] 예기치 못한 출력 타입: {type(output)}")

    da_seg = cv2.resize(da_seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    ll_seg = cv2.resize(ll_seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return da_seg, ll_seg

def extract_centerline_from_mask(mask):
    height, width = mask.shape
    roi = mask[int(height * 0.6):, :]  # 하단 40%만 사용

    left_points, right_points = [], []
    for y in range(roi.shape[0]):
        row = roi[y]
        xs = np.where(row > 0)[0]
        if len(xs) >= 2:
            left_x = xs[0]
            right_x = xs[-1]
            left_points.append((left_x, y + int(height * 0.6)))
            right_points.append((right_x, y + int(height * 0.6)))

    center_points = []
    for (lx, y1), (rx, y2) in zip(left_points, right_points):
        cx = int((lx + rx) / 2)
        center_points.append((cx, y1))
    return center_points, left_points, right_points

def calculate_angle_from_centerline(centerline_points):
    if len(centerline_points) < 10:
        return 0.0

    head = np.mean(centerline_points[:5], axis=0)
    tail = np.mean(centerline_points[-5:], axis=0)
    dx = tail[0] - head[0]
    dy = tail[1] - head[1]
    angle = np.arctan2(dy, dx)
    return angle

def pid_control(angle, kp=1.0):
    steer = kp * angle
    steer = max(min(steer, 1.0), -1.0)
    return steer

def apply_control(steering):
    global prev_steer

    if steering < -0.1:
        pyautogui.keyDown('a')
        pyautogui.keyUp('d')
    elif steering > 0.1:
        pyautogui.keyDown('d')
        pyautogui.keyUp('a')
    else:
        pyautogui.keyUp('a')
        pyautogui.keyUp('d')

    pyautogui.keyDown('w')  # 항상 전진
    prev_steer = steering
