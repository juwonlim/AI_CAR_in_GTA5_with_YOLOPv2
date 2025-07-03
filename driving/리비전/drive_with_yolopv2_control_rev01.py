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
#import pyautogui
import pydirectinput
import pydirectinput as pyautogui
pydirectinput.FAILSAFE = False

from yolopv2_inference.utils.utils import letterbox, split_for_trace_model, non_max_suppression, \
    driving_area_mask, lane_line_mask
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"E:/gta5_projectOCRtesseract/tesseract.exe"
sys.path.append(os.path.abspath("E:/gta5_project/AI_GTA5_Lanenet_Yolov2_Version/lanenet_inference"))



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
    


    
#yolopv2에 신규추가 (기존 calculate_steering_from_fit 대체??)

def calculate_offset_from_lane_mask(mask):
    h, w = mask.shape                        # 마스크의 높이, 너비 얻기
    midline = w // 2                         # 화면의 수평 중앙선 계산
    lane_indices = np.where(mask[h//2] == 1)[0]  # 화면 수직 절반 높이에서 차선 픽셀들의 x좌표 추출
    if len(lane_indices) == 0:              # 차선이 검출되지 않았을 경우
        return None                         # None 리턴하여 제어 중지 유도
    lane_center = int(np.mean(lane_indices)) # 검출된 차선 x좌표들의 평균값 → 중심 추정
    return lane_center - midline            # 중심점과 실제 중앙의 차이값 (offset)




#동일함
def apply_control(offset, threshold=10, slow_down_zone=20, max_speed_kmh=60):
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

    if current_speed is not None and current_speed < max_speed_kmh and abs(offset) < slow_down_zone:
        if frame_count % 10 == 0:
            pyautogui.keyDown('w')
        elif frame_count % 10 == 3:
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



#동일함
def stop_control():
    pyautogui.keyUp('w')
    pyautogui.keyUp('a')
    pyautogui.keyUp('d')




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




#신규 if 문
if __name__ == '__main__':
    import pyautogui
    import pydirectinput
    pydirectinput.FAILSAFE = False
    main_drive_loop()






