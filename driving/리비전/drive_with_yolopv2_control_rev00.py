
#lanenet 적용시 사용하던 모듈들 (삭제예정)
#from test_lanenet_final_rev12 import main_autonomous_loop
#from preprocess_for_lanenet import grab_screen
#from preprocess_for_lanenet import grab_speed_region
#import lanenet_inference.lanenet_model.lanenet as lanenet
#import lanenet_inference.lanenet_model.lanenet_postprocess as lanenet_postprocess
#import yaml
#from easydict import EasyDict as edict
#with open("E:/gta5_project/AI_GTA5_Lanenet_Yolov2_Version/lanenet_inference/config/tusimple_lanenet.yaml", 'r', encoding='utf-8') as f:
#    CFG = edict(yaml.safe_load(f))




import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import time
import numpy as np
import tensorflow as tf
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
    


#기존 calculate_steering_From_fit함수(lanenet 적용할떄 사용)
""" 
def calculate_steering_from_fit(fit_params, image_width=1280):
    if not fit_params:
        print("[WARNING] 차선을 감지하지 못함 - 정지")
        return None

    height = 720
    #차선이 2개일 경우 스터어링 계산
    if len(fit_params) == 2:
        left_x = fit_params[0][0] * height ** 2 + fit_params[0][1] * height + fit_params[0][2]
        right_x = fit_params[1][0] * height ** 2 + fit_params[1][1] * height + fit_params[1][2]
        lane_center = (left_x + right_x) / 2
    
    #차선이 하나만 있어도 스터어링 계산 시도함
    elif len(fit_params) == 1:
        one_x = fit_params[0][0] * height ** 2 + fit_params[0][1] * height + fit_params[0][2]
        lane_center = one_x - 200
    else:
        return None
    
    image_center = image_width / 2
    steering = lane_center - image_center

    print(f"[DEBUG] fit_params: {fit_params}")
    print(f"[DEBUG] computed steering: {steering}")


    
    return steering

"""


    
#yolopv2에 신규추가 (기존 calculate_steering_from_fit 대체??)
def calculate_offset_from_lane_mask(mask):
    h, w = mask.shape
    midline = w // 2
    lane_indices = np.where(mask[h//2] == 1)[0]
    if len(lane_indices) == 0:
        return None
    lane_center = int(np.mean(lane_indices))
    offset = lane_center - midline
    return offset



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



#기존 main_drive_loop함수 (lanenet적용시 사용)
''' 
def main_drive_loop(weights_path):
    print("[INFO] 자율주행 시작 (Y/N 전환, ESC 종료)")

  
    while True:
        controller.check_key_events()

        if controller.is_exit_pressed():
            print("[INFO] ESC 입력 - 자율주행 종료")
            stop_control()
            break

        if controller.is_auto_drive_enabled():
            result = main_autonomous_loop(weights_path)
            #offset = calculate_steering_from_fit(result['fit_params']) #이렇게 두면 한번만 차선인식후 정지가능성
            
            #아래처럼 차선없을 떄도 처리하게 둠 (차선이 인식이 안되면  cv2창이 리프레쉬 인되는 오류 해결위해)
            if result is None or result['fit_params'] is None:
                print("[WARNING] 차선을 감지하지 못함 - 정지")
                apply_control(None)
                continue
            offset = calculate_steering_from_fit(result['fit_params'])
            apply_control(offset)
                       
        else:
            stop_control()

        time.sleep(0.05)

 '''


 #test_lanenet_final_revxx.py파일에 있떤 main_drive_loop함수
""" 
 def main_autonomous_loop(weights_path):
    # 초기화: 모델 체크포인트 로딩은 1회만 수행되도록
    if not hasattr(main_autonomous_loop, "initialized"):
        saver.restore(sess=sess, save_path=weights_path)
        print("[DEBUG] Model checkpoint restored successfully.")
        main_autonomous_loop.initialized = True

    # 화면 캡처 및 전처리
    screen = grab_screen()
    image_vis = cv2.resize(screen, (1280, 720))  # 시각화용 원본
    image = enhance_contrast(image_vis)  # 대비 향상
    #image = image_vis.copy()
    image = cv2.resize(image, (512, 256))  # 모델 입력 크기
    image = image / 127.5 - 1.0

    # 네트워크 추론
    binary_seg_image, instance_seg_image = sess.run(
        [net.binary_seg, net.instance_seg],
        feed_dict={input_tensor: [image]}
    )

    # binary_segmentation 결과 이진화
    binary_seg_image = (binary_seg_image > 0.5).astype(np.uint8)

    print("[DEBUG] binary_seg_image.shape:", binary_seg_image.shape)

    # 후처리
    postprocess_result = postprocessor.postprocess(
        binary_seg_result=binary_seg_image[0],
        instance_seg_result=instance_seg_image[0],
        source_image=image_vis,
        with_lane_fit=True,
        data_source='tusimple'
    )

    mask_image = postprocess_result['mask_image']
    lane_params = postprocess_result['fit_params']

    # 시각화 (이미지에서와 동일하게 유지)
    cv2.namedWindow("mask_image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("mask_image", 640, 360)

    if mask_image is None:
        print("[WARNING] mask_image is None. Skipping display.")
        return {
            'fit_params': None,
            'mask_image': None,
            'source_image': image_vis
        }


    cv2.imshow("mask_image", mask_image)
    cv2.moveWindow("mask_image", 1280, 450)

    cv2.namedWindow("binary_seg", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("binary_seg", 640, 360)
    cv2.imshow("binary_seg", (binary_seg_image[0] * 255).astype(np.uint8))
    cv2.moveWindow("binary_seg", 1280, 0)
    cv2.waitKey(1)

        # instance_seg_image 시각화 추가
    embedding = instance_seg_image[0]
    for i in range(embedding.shape[2]):
        embedding[:, :, i] = cv2.normalize(embedding[:, :, i], None, 0, 255, cv2.NORM_MINMAX)

    embedding_vis = np.array(embedding, dtype=np.uint8)
    embedding_vis = embedding_vis[:, :, (2, 1, 0)] if embedding_vis.shape[2] == 3 else embedding_vis

    cv2.namedWindow("instance_seg", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("instance_seg", 640, 360)
    cv2.imshow("instance_seg", embedding_vis)
    cv2.moveWindow("instance_seg", 600, 750)




    # DEBUG 출력 (lane_params 확인용)
    if lane_params is not None:
        print("[DEBUG] lane_params:", lane_params)
    else:
        print("[WARNING] No lane detected.")
        return {'fit_params': None, 'mask_image': None, 'source_image': image_vis}

    # 결과 반환
    return {
        'mask_image': mask_image,
        'fit_params': lane_params,
        'source_image': image_vis
    }

"""




def main_drive_loop():
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

            da_seg_mask = driving_area_mask(seg)
            ll_seg_mask = lane_line_mask(ll)

            offset = calculate_offset_from_lane_mask(ll_seg_mask)
            apply_control(offset)

            # 선택적으로 시각화 (가상 중심선)
            output_img = draw_virtual_centerline(visual_img.copy(), offset)
            cv2.imshow("YOLOPv2 Drive View", output_img)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            stop_control()
            time.sleep(0.05)



#기존 if 문
""" 
if __name__ == '__main__':
    weights_path = 'E:/gta5_project/AI_GTA5_Lanenet_Yolov2_Version/lanenet_inference/lanenet_maybeshewill/tusimple_lanenet.ckpt'
    main_drive_loop(weights_path)
"""


#신규 if 문
if __name__ == '__main__':
    import pyautogui
    import pydirectinput
    pydirectinput.FAILSAFE = False
    main_drive_loop()