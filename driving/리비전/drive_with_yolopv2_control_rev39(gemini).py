
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


import pydirectinput
import pydirectinput as pyautogui
pydirectinput.FAILSAFE = False
import torch
from keyboard_input_only_rev00 import controller

from yolopv2_inference.utils.utils import letterbox, split_for_trace_model, non_max_suppression, \
    driving_area_mask, lane_line_mask

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


########################################################################################################################################

#내가 추가한 모듈들
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import torch
from loguru import logger
#from yolop.utils.visualize import vis
#from yolop.data.datasets.transform import ValTransform
import numpy as np
from yolopv2_inference.utils.utils import letterbox

#아래 내가 추가한 경로
weight_path = 'E:/gta5_project/AI_CAR_in_GTA5_with_YOLOPv2/yolopv2_inference/data/weights/yolopv2.pt' #이렇게 하면 리스트 형태로 전달됨,torch.jit.load는 리스트 대신 문자열 경로가필요함
sample_image = 'E:/gta5_project/AI_CAR_in_GTA5_with_YOLOPv2/sample_images/8.jpg'


import argparse
import time
from pathlib import Path






# Conclude setting / general reprocessing / plots / metrices / datasets
#yolopv2_inference.utils.utils --> 내가 수정한 경로
from yolopv2_inference.utils.utils import \
    time_synchronized,select_device, increment_path,\
    scale_coords,xyxy2xywh,non_max_suppression,split_for_trace_model,\
    driving_area_mask,lane_line_mask,plot_one_box,show_seg_result,\
    AverageMeter,\
    LoadImages





####################################여기서부터 DEMO.PY원본코드 ####################################

'''

def make_parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--weights', nargs='+', type=str, default='data/weights/yolopv2.pt', help='model.pt path(s)')
    #parser.add_argument('--weights', nargs='+', type=str, default='weight_path', help='model.pt path(s)') --> 문법 오류
    parser.add_argument('--weights', nargs='+', type=str, default=[weight_path], help='model.pt path(s)') #nargs='+' 는 --weights가 여러 개의 값을 받을 수 있도록 리스트로 처리,그런데 실제로는 하나만 전달했기 때문에, 리스트 안에 단일 문자열이 들어 있는 형태
    #parser.add_argument('--source', type=str, default='data/example.jpg', help='source')  # file/folder, 0 for webcam
    #parser.add_argument('--source', type=str, default='sample_image', help='source')  # file/folder, 0 for webcam --> 문법 오류
    parser.add_argument('--source', type=str, default=sample_image, help='source') 
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parser

'''



##원본 demo.py파일에 있던 detect함수 (참고용)
def detect():
    # setting and directories
    source, weights,  save_txt, imgsz = opt.source, opt.weights,  opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    # Load model
    stride =32
    model  = torch.jit.load(weights)  
    
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = model.to(device)
    

    if half:
        model.half()  # to FP16  
    model.eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        [pred,anchor_grid],seg,ll= model(img)
        t2 = time_synchronized()

        # waste time: the incompatibility of  torch.jit.trace causes extra time consumption in demo version 
        # but this problem will not appear in offical version 
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred,anchor_grid)
        tw2 = time_synchronized()

        # Apply NMS
        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
          
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img :  # Add bbox to image
                        plot_one_box(xyxy, im0, line_thickness=3)

            # Print time (inference)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            show_seg_result(im0, (da_seg_mask,ll_seg_mask), is_demo=True)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            #w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            #h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            w,h = im0.shape[1], im0.shape[0]
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    inf_time.update(t2-t1,img.size(0))
    nms_time.update(t4-t3,img.size(0))
    waste_time.update(tw2-tw1,img.size(0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')


#############################위는 demo.py의 핵심원본 코드들 , 건드리지말것#########################





def get_current_speed_from_screen():
    # 임시 속도 반환 (OCR 기능 미구현 상태)
    return 30.0



# === 가상 핸들 라인 시각화 함수 ===
def draw_handle_virtual_line(img, handle_angle):
    h, w = img.shape[:2]
    length = 200
    center = (w // 2, h)
    angle = -handle_angle
    x_end = int(center[0] + length * math.sin(angle))
    y_end = int(center[1] - length * math.cos(angle))
    cv2.line(img, center, (x_end, y_end), (255, 255, 255), 2)





# === 조향 제어 함수 ===

import time
import keyboard


def apply_control(steering):
    #pyautogui.keyDown('w')  -- detect_realtime함수에도 keydown있어서 차가 안나가게 만드는 원인
    if steering > 0.2:
        pyautogui.keyDown('d'); pyautogui.keyUp('a')
    elif steering < -0.2:
        pyautogui.keyDown('a'); pyautogui.keyUp('d')
    else:
        pyautogui.keyUp('a'); pyautogui.keyUp('d')




""" 
def apply_control(steering):
    print("[INFO] 주기적 가속 루프 시작 (3초 가속 / 0.5초 정지 반복)")
    try:
        while True:
            # 1. W 키 누르고 3초간 유지
            keyboard.press('w')
            print("[INFO] W 키 누름 - 가속 중")
            time.sleep(3.0)

            # 2. W 키에서 손 떼고 0.5초 쉬기
            keyboard.release('w')
            print("[INFO] W 키 뗌 - 정지 대기")
            time.sleep(0.5)
    except KeyboardInterrupt:
        keyboard.release('w')
        print("[INFO] 루프 종료됨. W 키 해제 완료.")

"""

def stop_control():
    pyautogui.keyUp('w')
    pyautogui.keyUp('a')
    pyautogui.keyUp('d')

# === 중심선 추출 함수 ===
def extract_centerline_from_mask(mask, output_img=None):
    h, w = mask.shape
    scan_rows = np.linspace(int(h * 0.3), h - 1, 50).astype(int) #h * 0.5 → 이미지의 하단 50%부터만 스캔함 ,50개의 가로 라인에서 차선을 검색 → 중간~하단만 사용하여 노이즈 줄이고 조향 반응 안정화
                                                                #0.3(스캔범위 위로 넓힘?) ,이렇게 스캔 범위를 위쪽으로 넓히면 더 많은 차선 포인트를 얻을 수 있어,left_points, right_points, centerline 생성에 도움
                                                                #특히 멀리 있는 차선이 얇고 희미하게 검출된 경우라면, 아래쪽보다 위쪽의 라인이 더 선명한 경우도 많음
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
    for y in scan_rows:
        lx = np.polyval(np.polyfit([y for x, y in left_points], [x for x, y in left_points], 1), y) if len(left_points) >= 2 else None
        rx = np.polyval(np.polyfit([y for x, y in right_points], [x for x, y in right_points], 1), y) if len(right_points) >= 2 else None
        #여기서부터 한쪽 차선만 있어도 중심선 만들어주는 코드
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
        for (x, y) in left_points + right_points:
            cv2.circle(output_img, (x, y), 2, (0, 0, 255), -1)  # 빨간 점 (양쪽 차선) ,여기서 output_img에 left/right points를 표시

    if output_img is not None:
        for x, y in center_points:
            cv2.circle(output_img, (x, y), 2, (0, 255, 255), -1)

    print(f"[DEBUG] left_points: {len(left_points)}, right_points: {len(right_points)}, centerline: {len(center_points)}")
    #return center_points #이건 tuple반환이 아님 - 에러 가능성 높음
    return center_points, left_points, right_points


# === 중심선 각도 계산 함수 ===
#도로 좌우 커브 상황에서 중심선 벡터 보정 적용
def calculate_angle_from_centerline(centerline_points):
    if len(centerline_points) < 2:
        return 0.0
    x1, y1 = centerline_points[0]
    x2, y2 = centerline_points[-1]
    #여기서부터 중심선 벡터보정
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return 0.0
    angle = math.atan2(dy, dx)
    return angle

# === PID 제어 함수 (단순 비례) ===
def pid_control(error, kp=1.0):
    return max(min(kp * error, 1.0), -1.0)


#########키 감지 함수 #######################
import threading
key_states = {"h": False, "y": False, "n": False}
def monitor_keys():
    while True:
        if keyboard.is_pressed("h"):
            key_states["h"] = True
        if keyboard.is_pressed("y"):
            key_states["y"] = True
        if keyboard.is_pressed("n"):
            key_states["n"] = True
        time.sleep(0.05)


# === 전역 변수 === 버전 25부터
handle_angle_zero = None  # H키 누를 때의 핸들 각도 기준
handle_angle = 0.0        # 현재 가상 핸들 각도
handle_angle = max(min(handle_angle, 90), -90) #너무 오래 누적되면 ±90도 이상 각도도 나올 수 있으니, 이처럼 제한 걸 수도 있음
auto_mode = False         # Y키로 자율주행 시작 여부
prev_steer = 0.0          # 저역 필터용

frame_count = 0  # apply_control() 외부에서 선언 필요

# === 가속 제어를 위한 전역 변수 ===
w_pressed = False
w_press_start = 0.0


def detect_realtime(model, device):
    global w_pressed, w_press_start  # ← 이거 추가


     # 초기 설정
    weights, imgsz = opt.weights, opt.img_size
    save_img = not opt.nosave
    stride = 32

    model  = torch.jit.load(weights[0])
    device = select_device(opt.device)
    half = device.type != 'cpu'
    model = model.to(device)
    if half:
        model.half()
    model.eval()
     # warm-up
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))


    # detect_realtime() 함수 시작 직전에
    threading.Thread(target=monitor_keys, daemon=True).start()


    print("[INFO] 자율주행을 시작하려면 H키로 핸들 정렬 후 Y키를 누르세요.")
    handle_initialized = False
    running = False
    global handle_angle
    handle_angle = 0.0  # 초기화
    imgsz = 640  # 전역 변수 선언

    while True:

        #시각화 처리
        start = time.time()
        screen = grab_screen()
        print("[INFO] Frame Processing Time:", time.time() - start)
        if screen is None or screen.size == 0:
            continue
        
         # 1. 시각화 및 제어의 기준이 될 이미지 (좌표계의 기준)
        vis_img = cv2.resize(screen, (1280, 720))
        draw_img = vis_img.copy() # ★★★ 핵심 수정 1: 시각화용 이미지를 여기서 복사합니다.


          # ========== [1] H, Y, N 키 우선 처리 ==========
        if key_states["h"]:
            handle_initialized = True
            print("[INFO] 핸들 정렬 기준선 설정됨 (0도).")
            #cv2.line(draw_img, (320, 480), (320, 240), (255, 255, 255), 2)
            cv2.line(draw_img, (640, 720), (640, 520), (255, 255, 255), 2) #현재 draw_img는 1280x720이므로 정중앙은 640
            key_states["h"] = False  # 다시 False로 초기화



        if key_states["y"] and handle_initialized:
            running = True
            start_time = time.time()  # 여기에 위치해야 함
            print("[INFO] 자율주행 시작!")
            key_states["y"] = False

        if key_states["n"]:
            running = False
            stop_control()
            print("[INFO] 자율주행 중지됨")
            key_states["n"] = False

        ''' 
        if running:
            # detect_realtime 내부 while 루프 안에서
            if time.time() - start_time < 60.0:
                pyautogui.keyDown('w')  # 60초간 천천히 가속
                print("[INFO] 초기 가속 중...")
            else:
                pyautogui.keyUp('w')    # 이후 관성 유지
        '''
         ###여기서 부터 3초가속###
        
        '''
        if running:
            now = time.time()

            if w_pressed:
                if now - w_press_start >= 3.0:
                    #keyboard.release('w')
                    pyautogui.keyUp('w')  # ← 수정: keyboard.release → pyautogui.keyUp
                    print("[INFO] W 키 뗌 - 정지 대기")
                    w_pressed = False
                    w_press_start = now
            else:
                if now - w_press_start >= 0.5:
                    #keyboard.press('w')
                    pyautogui.keyDown('w')  # ← 수정: keyboard.press → pyautogui.keyDown
                    print("[INFO] W 키 누름 - 가속 중")
                    w_pressed = True
                    w_press_start = now

        ####여기까지 3초 가속후 잠깐 쉬기#####   
        '''

        ''' 
        #gemini가 준것 ,계속 가속
        if running:
            pyautogui.keyDown('w') # 자율주행 중에는 항상 'w' 키를 누른 상태로 유지
            # 필요하다면 다른 제어 (브레이크, 조향) 로직 추가
        else:
            # 자율주행이 중지되면 모든 키를 뗀다.
            pyautogui.keyUp('w')
            pyautogui.keyUp('s')
            pyautogui.keyUp('a')
            pyautogui.keyUp('d')

        '''

        # 한 번만 누르고 유지하도록 로직을 분리
        if running:
            now = time.time()

            if w_pressed:
                # 3초 이상 가속하면 W 키 떼기
                if now - w_press_start >= 3.0:
                    pyautogui.keyUp('w')
                    print("[INFO] W 키 뗌 - 정지 대기")
                    w_pressed = False
                    w_press_start = now
            else:
                # 1초 대기 후 다시 W 키 누르기
                if now - w_press_start >= 1.0:
                    pyautogui.keyDown('w')
                    print("[INFO] W 키 누름 - 가속 중")
                    w_pressed = True
                    w_press_start = now


        

        # 2. 모델 입력용 이미지 전처리
        model_input = letterbox(vis_img, imgsz, stride=32)[0]
        model_input = model_input[:, :, ::-1].transpose(2, 0, 1)
        model_input = np.ascontiguousarray(model_input)
        img_tensor = torch.from_numpy(model_input).to(device)
        img_tensor = img_tensor.half() if half else img_tensor.float()
        img_tensor /= 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)


         # 3. YOLOPv2 추론
        [pred, anchor_grid], seg, ll = model(img_tensor)



         # 4. 후처리 및 마스크 추출
        pred = split_for_trace_model(pred, anchor_grid)
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        da_seg_mask = driving_area_mask(seg) # (720, 1280) 크기 마스크
        ll_seg_mask = lane_line_mask(ll)   # (720, 1280) 크기 마스크

        # 얇은 차선을 확장하여 인식률 향상 (차선 중심선 추출이 안되는 것으로 보여서 아래 2줄 추가)
        kernel = np.ones((5, 5), np.uint8)
        ll_seg_mask = cv2.dilate(ll_seg_mask.astype(np.uint8), kernel, iterations=1)


        # 5. 객체 감지 결과 시각화
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], vis_img.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    plot_one_box(xyxy, draw_img, line_thickness=2) # draw_img에 그립니다.

        # ... (H, Y, N 키 처리 로직은 그대로 유지) ...

        

           
 
            

            if ll_seg_mask is None or ll_seg_mask.shape[0] == 0:
                    print("[WARNING] 차선 마스크를 얻지 못했습니다. 건너뜁니다.")
                    continue

            # 중심선 추출 및 시각화 (이제 draw_img(1280, 720)에 정확히 그려집니다)
            centerline, left_points, right_points = extract_centerline_from_mask(ll_seg_mask, draw_img)
            if not centerline:
                print("[WARNING] 중심선을 만들 수 없음 → 조향 중단")
                stop_control()
                continue

            # 차선 및 중심선 시각화
            if len(centerline) >= 2:
                cv2.polylines(draw_img, [np.array(centerline, dtype=np.int32)], isClosed=False, color=(0, 255, 255), thickness=3) # 노란색 중심선
            if len(left_points) >= 2:
                cv2.polylines(draw_img, [np.array(left_points, dtype=np.int32)], isClosed=False, color=(255, 0, 0), thickness=2) # 파란색 좌측 차선
            if len(right_points) >= 2:
                cv2.polylines(draw_img, [np.array(right_points, dtype=np.int32)], isClosed=False, color=(0, 0, 255), thickness=2) # 빨간색 우측 차선          
                stop_control() 
                continue
            # 조향 각도 계산 및 제어
            angle_to_lane = calculate_angle_from_centerline(centerline)
            angle_error = angle_to_lane - handle_angle
            steer = pid_control(angle_error, kp=0.8) # kp값은 주행하며 튜닝 필요
            apply_control(steer)    

            # 핸들 각도 업데이트 (로직은 그대로 유지)
            handle_angle += steer * 5.0 
            handle_angle = max(min(handle_angle, math.radians(90)), math.radians(-90))
            print(f"[DEBUG] 조향입력: {math.degrees(steer):.2f}°, 누적핸들: {math.degrees(handle_angle):.2f}°")
        

        # 최종 결과 창 출력
        #cv2.imshow("Lane Mask", ll_seg_mask * 255) # ll_seg_mask는 0과 1이므로 255를 곱해줘야 보입니다.
        #print(f"[DEBUG] Lane Mask dtype: {ll_seg_mask.dtype}")  # 확인해보니 Lane Mask dtype: int32
        #cv2.resizeWindow("Lane Mask", 1280, 720) # 이렇게 해야 좌측상단만 보이지 않고 전체화면, 그러나 창이 너무 큼
        
        # 1. 마스크에 255를 곱해 시각화용으로 변환 (흑백 마스크)
        lane_mask_vis = (ll_seg_mask * 255).astype(np.uint8)
        # 2. 창용 출력 이미지 크기 축소 (640x360)
        display_img = cv2.resize(lane_mask_vis, (640, 360))
        

        # 3. 시각화 원본 비율유지하며 리사이즈
        cv2.imshow("Lane Mask", display_img)
        display_draw_img = cv2.resize(draw_img, (640,360))
        cv2.imshow("YOLOPv2 Result", display_draw_img)
        #key = cv2.waitKey(1) & 0xFF  # ← 여기에 하나만
        cv2.resizeWindow("Lane Mask", 640, 360)
        cv2.resizeWindow("YOLOPv2 Result", 640, 360) # 창 크기를 원본 비율로
        cv2.moveWindow("Lane Mask", 1280, 100)
        cv2.moveWindow("YOLOPv2 Result", 1280, 600)

  

        # 종료 조건 확인
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q') or (key_states["n"]):
            stop_control()
            break
cv2.destroyAllWindows()








if __name__ == '__main__':
    # 모델 경로와 이미지 크기 직접 설정
    WEIGHTS_PATH = "E:/gta5_project/AI_CAR_in_GTA5_with_YOLOPv2/yolopv2_inference/data/weights/yolopv2.pt"
    IMGSZ = 640

    # 모델 초기화
    device = select_device('0') # '0' for cuda:0 or 'cpu'
    model = torch.jit.load(WEIGHTS_PATH).to(device)
    if device.type != 'cpu':
        model.half()  # FP16
    model.eval()

    # GPU 워밍업
    if device.type != 'cpu':
        model(torch.zeros(1, 3, IMGSZ, IMGSZ).to(device).type_as(next(model.parameters())))
    
    # 옵션 객체를 직접 생성 (argparse 대신, make_parser함수 대체)
    class SimpleOpt:
        def __init__(self):
            self.conf_thres = 0.3
            self.iou_thres = 0.45
            self.classes = None
            self.agnostic_nms = False
            self.nosave = True # 이미지를 저장할 필요 없으므로 True
            self.weights = [WEIGHTS_PATH]  # ← 추가 ,모델 가중치 경로, opt.weights[0] 등에서 접근함
            self.img_size = IMGSZ         # ← 추가, 	입력 이미지 크기,letterbox() 및 split_for_trace_model() 등에서 사용됨
            self.device = '0'  # ← 이 줄 추가 ,GPU 혹은 CPU 설정 ,select_device(opt.device) 호출 때문

            # 선택 (예방적 정의)
            self.name = 'exp'
            self.project = 'runs/detect'
            self.save_conf = False
            self.save_txt = False
            self.exist_ok = False



    opt = SimpleOpt()

    with torch.no_grad():
        detect_realtime(model, device) # 모델과 장치를 인자로 전달




















#### 아래는 원본 코드지만 수정 가함#####
'''
if __name__ == '__main__':
    opt =  make_parser().parse_args()
    print(opt)

    with torch.no_grad():
        detect_realtime(model,device)

'''










####detect_realtime 에러나던 함수 보존################################################
""" 
def detect_realtime(model, device):


     # 초기 설정
    weights, imgsz = opt.weights, opt.img_size
    save_img = not opt.nosave
    stride = 32

    model  = torch.jit.load(weights[0])
    device = select_device(opt.device)
    half = device.type != 'cpu'
    model = model.to(device)
    if half:
        model.half()
    model.eval()
     # warm-up
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))


    # detect_realtime() 함수 시작 직전에
    threading.Thread(target=monitor_keys, daemon=True).start()


    print("[INFO] 자율주행을 시작하려면 H키로 핸들 정렬 후 Y키를 누르세요.")
    handle_initialized = False
    running = False
    global handle_angle
    handle_angle = 0.0  # 초기화
    imgsz = 640  # 전역 변수 선언

    while True:
        screen = grab_screen()
        frame_resize = screen.copy() #아래에 get_segmentation_masks(model, frame, device) 에서 사용됨
        frame_1280 = cv2.resize(frame_resize, (1280,720))
        frame = cv2.resize(frame_1280.copy(), (640, 640))
        #frame = letterbox(frame_1280, neww_shape=640)[0]

        # 시각화용 이미지 준비
        vis_img = cv2.resize(screen, (1280, 720))
         
         # 1. 모델 입력 전처리 (rev04 형식 그대로)
        model_input = letterbox(vis_img, imgsz, stride=32)[0]
        model_input = model_input[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
        model_input = np.ascontiguousarray(model_input)

        img_tensor = torch.from_numpy(model_input).to(device)
        img_tensor = img_tensor.half() if half else img_tensor.float()
        img_tensor /= 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        

        # 2. Yolo추론
        #t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model(img_tensor)
        #t2 = time_synchronized()


         # 3. 후처리
        #tw1 = time_synchronized() # waste time 측정
        pred = split_for_trace_model(pred, anchor_grid)
        #tw2 = time_synchronized() #waste time 측정

        #t3 = time_synchronized() #성능측정에만 사용
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   classes=opt.classes, agnostic=opt.agnostic_nms)
        #t4 = time_synchronized() #이것도 성능측정

        # 3. 마스크 추출
        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)


        # 3. 마스크 추출
        #da_seg_mask = seg.argmax(1)[0].byte().cpu().numpy()
        #ll_seg_mask = ll.argmax(1)[0].byte().cpu().numpy()

         # 4. 결과 처리 및 시각화
        for i, det in enumerate(pred):
            #s = '%gx%g ' % img_tensor.shape[2:]  --> s 변수는 원래 print() 출력용 문자열,지금은 print(f'{s}Done. (...)에만 쓰이고 있으므로, 줄여도 무방
            if len(det):
                #det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0.shape).round()
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], vis_img.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    #plot_one_box(xyxy, im0, line_thickness=2)
                    plot_one_box(xyxy, vis_img, line_thickness=2)

            #print(f'{s}Done. ({t2 - t1:.3f}s)') #위의 S변수 주석처리에 따라 이것도 주석
            #print(f'Done. ({t2 - t1:.3f}s)')




        #model_input = letterbox(vis_img_resize, new_shape=640)[0] #YOLO 입력용, #new shape 이거 안해주면 화면이 왼쪽 상단만 나옴, yolo는 정사각형 입력을 요구하므로 new_shape=640으로 모델용 이미지 생성
        draw_img_size = model_input.copy() #이건 numpy배열선언
        
        # 시각화용 별도 이미지 (왜곡 없이)
        #draw_img = vis_img_resize.copy()
        #draw_img = draw_img_size() # 이건 함수처럼 호출 이렇게 되면 에러 발생 --TypeError: 'numpy.ndarray' object is not callable
        #draw_img = draw_img_size.copy() #cv2.error: OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\highgui\src\window_w32.cpp:124: error: (-215:Assertion failed) bmi && width >= 0 && height >= 0 && (bpp == 8 || bpp == 24 || bpp == 32) in function 'FillBitmapInfo'
                                        # 이 에러 발생가능 코드

        draw_img = model_input.transpose(1, 2, 0)[:, :, ::-1].copy() #이렇게 해야 opencv가 받아들일수 있음.

        
        #vis_img, _, _, _ = letterbox(vis_img_resize, new_shape=640, auto=False, scaleFill=True, stride=32, return_params=True) #return_params=True를 사용하면 ratio, padding 등 좌표 보정값을 얻을 수 있어
                                                                                                                              #그걸 이용해 centerline, left_points, right_points 좌표를 보정해서 정확히 표시

       

          # ========== [1] H, Y, N 키 우선 처리 ==========
        if key_states["h"]:
            handle_initialized = True
            print("[INFO] 핸들 정렬 기준선 설정됨 (0도).")
            cv2.line(draw_img, (320, 480), (320, 240), (255, 255, 255), 2)
            key_states["h"] = False  # 다시 False로 초기화

        #if keyboard.is_pressed('y') and handle_initialized: # 이게 key_states['y'] 보다 먼저 실행되기 땜시, Y키룰 누르지 않아도 조건이 만족된 것으로 착각하고 바로 주행
            #running = True


        if key_states["y"] and handle_initialized:
            running = True
            start_time = time.time()  # 여기에 위치해야 함
            print("[INFO] 자율주행 시작!")
            key_states["y"] = False

        if key_states["n"]:
            running = False
            stop_control()
            print("[INFO] 자율주행 중지됨")
            key_states["n"] = False


        if running:
            # detect_realtime 내부 while 루프 안에서
            if time.time() - start_time < 4.0:
                pyautogui.keyDown('w')  # 4초간 천천히 가속
                print("[INFO] 초기 가속 중...")
            else:
                pyautogui.keyUp('w')    # 이후 관성 유지

        
            #da_seg, ll_seg = get_segmentation_masks(model, frame, device)
            # 수정 코드 (정확한 입력 이미지 사용)
            #da_seg, ll_seg = get_segmentation_masks(model, model_input, device) #현재 구조에서는 YOLO 입력용 이미지를 그대로 get_segmentation_masks()에도 전달해야 함
            
            
            if da_seg_mask is None or ll_seg_mask is None:
                print("[WARNING] YOLO 출력 비정상 → 건너뜀")
                continue

                 # Lane Mask 시각화
                 #lane_line_mask()는 정상적인 seg 출력을 받는 한 None을 반환하지 않기 때문에,실제로는 이 조건문 자체가 필요 없습
            ''' 
            if ll_seg_mask is None or ll_seg.shape[0] == 0:
                print("[ERROR] 잘못된 차선 마스크 → 건너뜀")
                continue
            '''
            print("[DEBUG] showing mask, shape:", ll_seg_mask.shape, " values:", np.unique(ll_seg_mask))
            print("[DEBUG] Lane mask visualization 시작") #해당 로그가 찍히지 않는다면 if running: 블록 안에서 흐름이 중간에 continue로 빠지고 있음
            cv2.imshow("Lane Mask", ll_seg_mask * 255)
            cv2.resizeWindow("Lane Mask", 640, 360)
            cv2.moveWindow("Lane Mask", 1280, 100)
            cv2.waitKey(1)

            # 중심선 추출 및 시각화
            centerline, left_points, right_points = extract_centerline_from_mask(ll_seg_mask, draw_img)

            # 선으로 (차선을) 시각화
            if len(centerline) >= 2:
                cv2.polylines(draw_img, [np.array(centerline, dtype=np.int32)], isClosed=False, color=(0, 255, 255), thickness=2)
            if len(left_points) >= 2:
                cv2.polylines(draw_img, [np.array(left_points, dtype=np.int32)], isClosed=False, color=(255, 0, 0), thickness=2)
            if len(right_points) >= 2:
                cv2.polylines(draw_img, [np.array(right_points, dtype=np.int32)], isClosed=False, color=(0, 0, 255), thickness=2)


            if not centerline:
                print("[WARNING] 차선 중심선(centerline) 추출 실패")
                continue
          
            #for (x, y) in centerline: --이거는 안쓰이나?
                #cv2.circle(vis_img, (int(x), int(y)), 2, (0, 255, 255), -1)

            # 조향 각도 계산 및 적용
            angle_to_lane = calculate_angle_from_centerline(centerline)
            angle_error = angle_to_lane - handle_angle
            steer = pid_control(angle_error)
            apply_control(steer)

            # 누적 핸들 각도 업데이트
            handle_angle += steer * 5.0
            handle_angle = max(min(handle_angle, math.radians(90)), math.radians(-90))

            print(f"[DEBUG] 조향입력: {math.degrees(steer):.2f}°, 누적핸들: {math.degrees(handle_angle):.2f}°")

           

            #일단 cv2창부터 열고 나서 중심선 계산, 이렇게 해서 mask창 강제로 open
            # 그 다음 중심선 계산
          
            # 차선 중심선 시각화 (아직까지 점으로 시각화 되던 코드)
            #centerline = extract_centerline_from_mask(ll_seg, draw_img) #이 함수에 draw_img자리가 원래  output_img자리였음 그러므로 draw_img에는 양쪽차선점(left_points, right_points)와 중심선(center line)이 그려지게 됨
            #for (x, y) in centerline:
            #    cv2.circle(draw_img, (x, y), 2, (0, 255, 255), -1) # 노란 점 --> 아작 까지는 차선이 점으로 처리

            
            



          

        # YOLO 결과 시각화
        # 결과 창에 출력
        cv2.imshow("YOLOPv2 Result", draw_img)
        #cv2.imshow("YOLOPv2 Result", vis_img)
        cv2.resizeWindow("YOLOPv2 Result", 640, 360)
        cv2.moveWindow("YOLOPv2 Result", 1280, 600)

        # 종료 조건
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()
'''





##아래는 폐기된 함수들
''' 
# === YOLOPv2 세그멘테이션 마스크 함수 ===
def get_segmentation_masks(model, img, device):
    model.eval()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device).half()

    with torch.no_grad():
        output = model(img_tensor)

    if isinstance(output, dict):
        da_seg = output['drivable'][0].argmax(0).byte().cpu().numpy()
        ll_seg = output['lane'][0].argmax(0).byte().cpu().numpy()
    elif isinstance(output, (list, tuple)):
        da_seg, ll_seg = output[:2] #여기서 ([pred, anchor_grid], seg)를 da_seg로 착각함 --> 	[pred, anchor_grid], seg, ll = model(img_tensor) 이렇게 수정해야함
        if isinstance(da_seg, list):
            da_seg = da_seg[0]
        if isinstance(ll_seg, list):
            ll_seg = ll_seg[0]
        da_seg = da_seg.argmax(0).byte().cpu().numpy() #위의 문제로 여기서 'AttributeError: 'tuple' object has no attribute 'argmax''이 문제가 발생함
        ll_seg = ll_seg.argmax(0).byte().cpu().numpy() #위의 문제로 여기서 'AttributeError: 'tuple' object has no attribute 'argmax''이 문제가 발생함
    else:
        return None, None

    return da_seg, ll_seg


#yolopv2 jit모델의 출력값에 맞춘 것 --> 그러나 cv2.error: (-215:Assertion failed) bmi && width >= 0 && height >= 0 && (bpp == 8 || bpp == 24 || bpp == 32) in function 'FillBitmapInfo' 에러발생
def get_segmentation_masks(model, img, device):
    model.eval()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device).half()

    with torch.no_grad():
        [pred, anchor_grid], seg, ll = model(img_tensor)  #정확한 분해 , YOLOPv2 JIT 모델은 출력값이 [pred, anchor_grid], seg, ll 형식의 튜플

    da_seg = seg.argmax(0).byte().cpu().numpy()
    ll_seg = ll.argmax(0).byte().cpu().numpy()

    return da_seg, ll_seg





def get_segmentation_masks(model, img, device):
    model.eval()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device).half()
    

    with torch.no_grad():
        output = model(img_tensor)

    # 정확하게 분해
    if isinstance(output, (list, tuple)) and len(output) == 3:
        _, seg, ll = output
    else:
        print("[ERROR] YOLOPv2 모델 출력값이 예상과 다름")
        return None, None

    if not isinstance(seg, torch.Tensor) or not isinstance(ll, torch.Tensor):
        print("[ERROR] seg 또는 ll이 torch.Tensor 타입이 아님")
        return None, None

    da_seg = seg.argmax(1)[0].byte().cpu().numpy()
    ll_seg = ll.argmax(1)[0].byte().cpu().numpy()

    return da_seg, ll_seg

"""