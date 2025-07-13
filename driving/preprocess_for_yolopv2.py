# === preprocess.py ===

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #이건 현재 파일 기준으로 상위 폴더(=루트) 를 자동으로 찾아주는 방식이야
import cv2
import numpy as np
from PIL import ImageGrab
import win32ui #'CreateDCFromHandle' 사용시 필요
import win32con #비트블릿 복사할 때 필요
import win32gui #ReleaseDC 할 때 필요







#win32gui를 이용해서 GTA5창 자동인식
def grab_screen():
    hwnd = win32gui.FindWindow(None, "Grand Theft Auto V")
    if hwnd == 0:
        print("[ERROR] GTA5 창을 찾을 수 없습니다.")
        return np.zeros((720, 1280, 3), dtype=np.uint8) #height = 720, width = 1280 크기의 **검정 화면(dummy image)**를 생성해서 이후 파이프라인이 완전히 망가지지 않도록 막기 위해 넣은 것, resize아님
                                                        #GTA5 꺼져 있을 때 대비한 더미 이미지
 

    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top

    hwindc = win32gui.GetWindowDC(hwnd)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()

    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (0, 0), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    img = img[..., :3]  # BGRA → BGR
    screen = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return screen #RGB로 리턴



#우측 하단의 속도계 검출로직 

"""
def grab_speed_region(screen):
    # 입력은 반드시 (720, 1280, 3) 사이즈여야 함
    if screen.shape[0] != 720 or screen.shape[1] != 1280:
        screen = cv2.resize(screen, (1280, 720))  # 안전장치

    speed_region = screen[650:710, 1100:1250]
    return speed_region

 """



def grab_speed_region(screen):
    # 입력은 반드시 (720, 1280, 3) 사이즈여야 함
    if screen.shape[0] != 720 or screen.shape[1] != 1280:
        screen = cv2.resize(screen, (1280, 720))  # 안전장치

    # 너비, 높이를 더 크게 설정하고 위로 살짝 이동
    x, y, w, h = 1180, 660, 100, 50
    return screen[y:y+h, x:x+w]



'''  
# preprocess_for_yolopv2.py 마지막에 추가
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # scale ratio (new / old) and compute padding
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    # resize
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, ratio, (dw, dh)
'''