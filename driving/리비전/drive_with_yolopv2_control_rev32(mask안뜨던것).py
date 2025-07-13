
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


# === 전역 변수 === 버전 25부터
handle_angle_zero = None  # H키 누를 때의 핸들 각도 기준
handle_angle = 0.0        # 현재 가상 핸들 각도
handle_angle = max(min(handle_angle, 90), -90) #너무 오래 누적되면 ±90도 이상 각도도 나올 수 있으니, 이처럼 제한 걸 수도 있음
auto_mode = False         # Y키로 자율주행 시작 여부
prev_steer = 0.0          # 저역 필터용

frame_count = 0  # apply_control() 외부에서 선언 필요



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
def apply_control(steering):
    pyautogui.keyDown('w')
    if steering > 0.2:
        pyautogui.keyDown('d'); pyautogui.keyUp('a')
    elif steering < -0.2:
        pyautogui.keyDown('a'); pyautogui.keyUp('d')
    else:
        pyautogui.keyUp('a'); pyautogui.keyUp('d')

def stop_control():
    pyautogui.keyUp('w')
    pyautogui.keyUp('a')
    pyautogui.keyUp('d')

# === 중심선 추출 함수 ===
def extract_centerline_from_mask(mask, output_img=None):
    h, w = mask.shape
    scan_rows = np.linspace(int(h * 0.5), h - 1, 50).astype(int) #h * 0.5 → 이미지의 하단 50%부터만 스캔함 ,50개의 가로 라인에서 차선을 검색 → 중간~하단만 사용하여 노이즈 줄이고 조향 반응 안정화
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
        for x, y in center_points:
            cv2.circle(output_img, (x, y), 2, (0, 255, 255), -1)

    return center_points

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
        da_seg, ll_seg = output[:2]
        if isinstance(da_seg, list):
            da_seg = da_seg[0]
        if isinstance(ll_seg, list):
            ll_seg = ll_seg[0]
        da_seg = da_seg.argmax(0).byte().cpu().numpy()
        ll_seg = ll_seg.argmax(0).byte().cpu().numpy()
    else:
        return None, None

    return da_seg, ll_seg



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





start_time = time.time()  # detect_realtime() 밖에서 정의  (차량 천천히 출발시키기 위한 조건 선언)

def detect_realtime(model, device):
    # detect_realtime() 함수 시작 직전에
    threading.Thread(target=monitor_keys, daemon=True).start()


    print("[INFO] 자율주행을 시작하려면 H키로 핸들 정렬 후 Y키를 누르세요.")
    handle_initialized = False
    running = False
    global handle_angle
    handle_angle = 0.0  # 초기화

    while True:
        screen = grab_screen()
        frame = screen.copy() #어따쓰게 스크린 복제?
        vis_img_resize = cv2.resize(screen, (1280, 720))
        vis_img = letterbox(vis_img_resize, new_shape=640)[0] #new shape 이거 안해주면 화면이 왼쪽 상단만 나옴, yolo는 정사각형 입력을 요구하므로 new_shape=640으로 모델용 이미지 생성

        # 속도계 영역 시각화
        #cv2.rectangle(vis_img, (1180, 660), (1280, 710), (255, 0, 0), 2)
        # 원하는 목표 속도 (km/h)
        #target_speed = 30  
        #current_speed = get_current_speed_from_screen() # 현재 속도 확인
        #print(f"[INFO] 현재 속도: {current_speed} km/h")

          # ========== [1] H, Y, N 키 우선 처리 ==========
        if key_states["h"]:
            handle_initialized = True
            print("[INFO] 핸들 정렬 기준선 설정됨 (0도).")
            cv2.line(vis_img, (320, 480), (320, 240), (255, 255, 255), 2)
            key_states["h"] = False  # 다시 False로 초기화

        if key_states["y"] and handle_initialized:
            running = True
            print("[INFO] 자율주행 시작!")
            key_states["y"] = False

        if key_states["n"]:
            running = False
            stop_control()
            print("[INFO] 자율주행 중지됨")
            key_states["n"] = False



        # detect_realtime 내부 while 루프 안에서
        if time.time() - start_time < 4.0:
            pyautogui.keyDown('w')  # 4초간 천천히 가속
            print("[INFO] 초기 가속 중...")
        else:
            pyautogui.keyUp('w')    # 이후 관성 유지


            # ========== [2] 현재 속도 체크는 그 다음에 ==========
        #current_speed = get_current_speed_from_screen()
        #print(f"[INFO] 현재 속도: {current_speed} km/h")
        

        ''' 
         # 개선안  current_speed가 None일 경우, 가속 유도
        if current_speed is None:
            pyautogui.keyDown('w')  # 정지 상태라도 일단 출발 시도
            print("[INFO] 속도 인식 실패 → 기본 가속")
            #return 0 # 이건 루프탈출
            continue

        if current_speed < 30 - 2:
            pyautogui.keyDown('w') # 가속
        else:
            pyautogui.keyUp('w') # 가속 멈춤 (관성 유지)
        '''
    

        



        #####여기서부터#######

        if running:
            da_seg, ll_seg = get_segmentation_masks(model, frame, device)
            if da_seg is None or ll_seg is None:
                print("[WARNING] YOLO 출력 비정상 → 건너뜀")
                continue

            # 중심선 추출 및 시각화
            centerline = extract_centerline_from_mask(ll_seg, vis_img)
            for (x, y) in centerline:
                cv2.circle(vis_img, (int(x), int(y)), 2, (0, 255, 255), -1)

            # 조향 각도 계산 및 적용
            angle_to_lane = calculate_angle_from_centerline(centerline)
            angle_error = angle_to_lane - handle_angle
            steer = pid_control(angle_error)
            apply_control(steer)

            # 누적 핸들 각도 업데이트
            handle_angle += steer * 5.0
            handle_angle = max(min(handle_angle, math.radians(90)), math.radians(-90))

            print(f"[DEBUG] 조향입력: {math.degrees(steer):.2f}°, 누적핸들: {math.degrees(handle_angle):.2f}°")

            # Lane Mask 시각화
            cv2.imshow("Lane Mask", ll_seg * 255)
            cv2.resizeWindow("Lane Mask", 640, 360)
            cv2.moveWindow("Lane Mask", 1280, 100)

        # YOLO 결과 시각화
        cv2.imshow("YOLOPv2 Result", vis_img)
        cv2.resizeWindow("YOLOPv2 Result", 640, 360)
        cv2.moveWindow("YOLOPv2 Result", 1280, 600)

        # 종료 조건
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    detect_realtime(model, device)










""" 
# === 메인 루프 ===
def main():
    global handle_angle
    print("[INFO] 자율주행을 시작하려면 H키로 핸들 정렬 후 Y키를 누르세요.")
    model = load_model(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    handle_initialized = False
    running = False

    while True:
        screen = grab_screen()
        vis_img = screen.copy()
        frame = screen.copy()

        # 가상 핸들 0도 기준선
        if handle_initialized:
            draw_handle_virtual_line(vis_img, handle_angle)

        if keyboard.is_pressed('h'):
            handle_initialized = True
            handle_angle = 0.0
            print("[INFO] 핸들 정렬 기준선 설정됨 (0도).")
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

        if running:
            da_seg, ll_seg = get_segmentation_masks(model, frame, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            if da_seg is None or ll_seg is None:
                continue

            centerline = extract_centerline_from_mask(ll_seg, vis_img)
            angle_to_lane = calculate_angle_from_centerline(centerline)
            angle_error = angle_to_lane - handle_angle
            steer = pid_control(angle_error)
            apply_control(steer)

            handle_angle += steer * 5.0
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



#detect()함수를 실시간용으로 개조했던 함수

""" 
def detect_realtime():
    # 초기 설정
    weights, imgsz = opt.weights, opt.img_size
    save_img = not opt.nosave
    stride = 32

    #save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    #save_dir.mkdir(parents=True, exist_ok=True)

    # 성능 측정기
    #inf_time, waste_time, nms_time = AverageMeter(), AverageMeter(), AverageMeter()

    # 모델 로드
    #model = torch.jit.load(weights) #torch.jit.load()는 **문자열(str)**을 기대함 , model = torch.jit.load("E:/path/to/yolopv2.pt") --> OK (문자열)
                                     #weights는 리스트 → 에러: list has no attribute 'read'
    model  = torch.jit.load(weights[0]) #AttributeError: 'list' object has no attribute 'read' (이는 torch.jit.load() 함수에 list 타입을 넘겼기 때문에 발생한 오류)
                                        #default=[weight_path] 로 설정하면 opt.weights는 list 타입이 됨
                                        #하지만 torch.jit.load()는 string 경로 하나만 받아야 함. 즉, torch.jit.load(opt.weights[0]) 이렇게 수정
                                        #이weights[0]으로 리스트의 첫 번째 요소(= 문자열 경로)를 꺼내야 함

    
    device = select_device(opt.device)
    half = device.type != 'cpu'
    model = model.to(device)
    if half:
        model.half()
    model.eval()

    # warm-up
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    t0 = time.time()

    while True:
        # 1. Grab screen from GTA5
        #screen = grab_screen(region=None) #preprocess의 grab_screen함수가 def grab_screen(): 이렇게 인자 없이 시작되기에 region이란게 여기 있으면 안됨
        # 1. 화면 캡처
        screen = grab_screen()

        if screen is None or screen.size == 0:
            print("grab_screen() 실패: 빈 이미지가 반환됨")
            continue  # 다음 루프로 넘어가서 다시 시도

        if screen is None or screen.shape[0] == 0 or screen.shape[1] == 0:
            print("grab_screen()에서 올바르지 않은 이미지 크기 반환. 루프를 계속합니다.")
            continue

        # 2. 시각화용 이미지 (Segmentation 결과 오버레이용)
        visual_img = cv2.resize(screen, (1280, 720))  # 혹은 모델 출력 해상도에 맞춤
        # 3. 모델 입력용 이미지 (letterbox로 정규화)
        #input_img = letterbox(visual_img, imgsz, stride=32)[0]
        #input_img = input_img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
        #input_img = np.ascontiguousarray(input_img)


        #im0 = cv2.resize(screen, (imgsz, imgsz))
        #input_img = cv2.resize(screen, (imgsz, imgsz)) #input_img.shape = (640, 640, 3)인데,color_mask.shape = (720, 1280) 이런 식으로 서로 크기가 안 맞음.
                                                        
        #img = letterbox(input_img, imgsz, stride=32)[0]
        model_input = letterbox(visual_img, imgsz, stride=32)[0]
        model_input = model_input[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
        model_input = np.ascontiguousarray(model_input)

        img_tensor = torch.from_numpy(model_input).to(device)
        img_tensor = img_tensor.half() if half else img_tensor.float()
        img_tensor /= 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # 2. 추론
        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model(img_tensor)
        t2 = time_synchronized()

        # 3. 후처리
        #tw1 = time_synchronized() # waste time 측정
        pred = split_for_trace_model(pred, anchor_grid)
        #tw2 = time_synchronized() #waste time 측정

        #t3 = time_synchronized() #성능측정에만 사용
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   classes=opt.classes, agnostic=opt.agnostic_nms)
        #t4 = time_synchronized() #이것도 성능측정

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # 4. 결과 처리 및 시각화
        for i, det in enumerate(pred):
            #s = '%gx%g ' % img_tensor.shape[2:]  --> s 변수는 원래 print() 출력용 문자열,지금은 print(f'{s}Done. (...)에만 쓰이고 있으므로, 줄여도 무방
            if len(det):
                #det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0.shape).round()
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], visual_img.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    #plot_one_box(xyxy, im0, line_thickness=2)
                    plot_one_box(xyxy, visual_img, line_thickness=2)

            #print(f'{s}Done. ({t2 - t1:.3f}s)') #위의 S변수 주석처리에 따라 이것도 주석
            print(f'Done. ({t2 - t1:.3f}s)')

            #show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True) #원본
            output_img = show_seg_result(visual_img, (da_seg_mask, ll_seg_mask), is_demo=True) #utils.py파일에서 show_seg_result함수의 return값을 img로 정해둠, 리턴값을 output_img에서 받음
            cv2.namedWindow("YOLOPv2 Result", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLOPv2 Result", 640, 360)
            cv2.moveWindow("YOLOPv2 Result", 1280, 150)
            cv2.imshow("YOLOPv2 Result", output_img)


        # 5. ESC 키 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == 27:
            print("종료 명령 수신. 루프를 종료합니다.")
            break

    print('완료. 총 처리 시간: (%.3fs)' % (time.time() - t0))
""" 

#########################################################여기까지 detect_realtim함수#################