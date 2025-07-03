import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #이건 현재 파일 기준으로 상위 폴더(=루트) 를 자동으로 찾아주는 방식

import cv2
import numpy as np
from darkflow.net.build import TFNet # YOLOv2를 불러오는 다크플로우 패키지
from shapely.geometry import box, Polygon # 물체 충돌 계산에 사용될 사각형(폴리곤) 클래스
from object_detection.direction import Direct # 주행 방향 enum





#roi = Polygon([(100, 440), (350, 250), (450, 250), (700, 440)]) #preprocess.py에서 받아온 roi를 안쓰고 여기서 다시 정의하는건데 1280x720이랑 안맞음

#내가 다시 정의한 것, 직접 그려서 확인해 본 것
roi = Polygon([(0, 0),(0, 550),(1280, 550),(1280, 0)])


# set YOLO options
""" 
options = {
    'model': 'cfg/yolo2.cfg', #원본 yolo를 yolo2로 수정함   ,# YOLO 모델 구조
    'load': 'yolov2.weights',  # 학습된 가중치
    'threshold': 0.3,  # 신뢰도 임계값
    'gpu': 0.5  # GPU 사용 비율
}
"""
options = {
    'model': os.path.join(os.path.dirname(__file__), 'yolov2.cfg'),
    'load': os.path.join(os.path.dirname(__file__), 'yolov2.weights'), #weight파일 경로가 하드코딩되어 있음. 즉, object_detect.py가 위치한 폴더에 weights와 cfg와 labels.txt가 존재해야한다는 의미
    'threshold': 0.3,
    'gpu': 0.2, #GPU점유낮춰서 도서관 사용시 소음감소
    'labels': os.path.join(os.path.dirname(__file__), 'labels.txt')
}


tfnet = TFNet(options) # YOLO 모델 초기화


# capture = cv2.VideoCapture('gta2.mp4')
# 랜덤 색상 / 검은색 셋
t = (0, 0, 0)
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
colors2 = [tuple(t) for j in range(15)]


# 교차로에서 신호등 색상 인식 함수
def light_recog(frame, direct, traffic_lights):
    traffic_light = traffic_lights[0] # 첫 번째 신호등으로 초기화

    # find out which traffic light to follow, if there are several
    # 여러 개의 신호등이 있는 경우, 주행 방향에 따라 판단할 신호등 선택
    if len(traffic_lights) > 1:
        # if we need to go to the right
        if direct == Direct.RIGHT or direct == Direct.SLIGHTLY_RIGHT:
            for tl in traffic_lights:
                if tl['topleft']['x'] > traffic_light['topleft']['x']:
                    traffic_light = tl # 가장 오른쪽 신호등 선택
        # straight or left
        # 직진 또는 좌회전
        else: 
            for tl in traffic_lights:
                if tl['topleft']['x'] < traffic_light['topleft']['x']:
                    traffic_light = tl # 가장 왼쪽 신호등 선택

    # coordinates of the traffic light
    # 선택된 신호등의 좌표 추출
    top_left = (traffic_light['topleft']['x'], traffic_light['topleft']['y'])
    bottom_right = (traffic_light['bottomright']['x'], traffic_light['bottomright']['y'])

    # crop the frame to the traffic light
     # 신호등 영역 추출 → HSV 색공간으로 변환
    roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    color_detected = ''

    # possible color ranges for traffic lights
     # HSV 색상 범위 정의
    red_lower = np.array([136, 87, 111], dtype=np.uint8)
    red_upper = np.array([180, 255, 255], dtype=np.uint8)

    yellow_lower = np.array([22, 60, 200], dtype=np.uint8)
    yellow_upper = np.array([60, 255, 255], dtype=np.uint8)

    green_lower = np.array([50, 100, 100], dtype=np.uint8)
    green_upper = np.array([70, 255, 255], dtype=np.uint8)

    # find what color the traffic light is showing
    # 마스크 생성: 각 색 범위에 해당하는 픽셀 검출
    red = cv2.inRange(hsv, red_lower, red_upper)
    yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green = cv2.inRange(hsv, green_lower, green_upper)

    kernel = np.ones((5, 5), np.uint8)   # 잡음 제거용 커널 정의

    # 마스크 후처리: 팽창 & 마스킹 이미지 추출
    red = cv2.dilate(red, kernel)
    res = cv2.bitwise_and(roi, roi, mask=red)
    green = cv2.dilate(green, kernel)
    res2 = cv2.bitwise_and(roi, roi, mask=green)

    # 신호등 색 판별: contour가 있으면 해당 색
    #(_, contours, hierarchy) = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #opencv 3.x스타일
    contours, hierarchy = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #opencv 4.x스타일

    for contour in enumerate(contours):
        color_detected = "Red"

    #(_, contours, hierarchy) = cv2.findContours(yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in enumerate(contours):
        color_detected = "Yellow"

    #(_, contours, hierarchy) = cv2.findContours(green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in enumerate(contours):
        color_detected = "Green"

 # 특정 영역에 있을 때만 텍스트 표시 (좌측 하단 HUD 방지용)
    if (0 <= top_left[1] and bottom_right[1] <= 437) and (244 <= top_left[0] and bottom_right[0] <= 630):
        frame = cv2.putText(frame, color_detected, bottom_right, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

# 일반적으로 신호등에 텍스트 표시
    frame = cv2.putText(frame, color_detected, bottom_right, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    return frame, color_detected # 수정된 이미지와 인식된 색상 반환

#TFNet(options)	YOLOv2 객체 감지 모델 로딩
#light_recog()	신호등이 "빨간불인지, 노란불인지, 초록불인지"를 인식해 반환함
#HSV 범위	색상 인식을 위한 기준 정의 (Red/Yellow/Green)


############################# 여끼까지 #1
# 차량과의 거리 계산 함수
def distance_to_car(frame, top_left, bottom_right):
    distance = None

    # myRoi_array= np.array([[(0, 490), (309, 269), (490, 270), (800,473)]])
    # process_img = region_of_interest(frame, myRoi_array)
    # cv2.imshow("precess_img", process_img)

    # roi = Polygon([(15, 472), (330, 321), (470, 321), (796, 495)])
    #roi = Polygon([(100, 470), (350, 280), (450, 280), (700, 470)]) # ROI: 우리가 관심 있는 충돌 영역 정의 (사다리꼴)
    #roi = Polygon([(100, 440), (350, 250), (450, 250), (700, 440)]) #preprocess.py에서 받아온 1280x450사이즈에 맞춰서 수정
    car = box(top_left[0], top_left[1], bottom_right[0], bottom_right[1])  # 차량이 포함된 사각형 정의 (YOLO bbox)


     # 충돌 구역 안에 차량이 들어왔는지 확인
    if roi.intersects(car):
        mid_x = (bottom_right[0] + top_left[0]) / 2
        mid_y = (top_left[1] + bottom_right[1]) / 2

         # 단순한 거리 추정 (가로폭 기반 계산)
        distance = round((1 - ((bottom_right[0] / 800) - (top_left[0] / 800))) ** 4, 1)
        
        # 이미지에 거리 표시
        frame = cv2.putText(frame, '{}'.format(distance), (int(mid_x), int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
        
        # 경고 메시지 표시
        cv2.putText(frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]],
                    'WARNING!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    return frame, distance



# 보행자와의 거리 계산 함수
def distance_to_human(frame, top_left, bottom_right):
    distance = None

    #roi = Polygon([(90, 470), (350, 280), (450, 280), (700, 470)])
    #roi = Polygon([(100, 440), (350, 250), (450, 250), (700, 440)])  #preprocess.py에서 받아온 1280x450사이즈에 맞춰서 수정
    person = box(top_left[0], top_left[1], bottom_right[0], bottom_right[1])

    if roi.intersects(person):
        mid_x = (bottom_right[0] + top_left[0]) / 2
        mid_y = (top_left[1] + bottom_right[1]) / 2
        
         # 보행자는 거리 민감도를 더 높게 설정 (지수 더 큼)
        distance = round((1 - ((bottom_right[0] / 800) - (top_left[0] / 800))) ** 15, 1)
        frame = cv2.putText(frame, '{}'.format(distance), (int(mid_x), int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
        cv2.putText(frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]],
                    'WARNING!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    return frame, distance


# YOLO로 객체 감지 + 각 객체에 대한 후처리 (신호등, 차량, 사람)
def yolo_detection(screen, direct):
    
    #processed = get_preprocessed(screen)  # preprocess.py로부터 이미지 전체 받아옴
    #screen = processed["resized"]  # YOLO에 적용할 적절한 크기의 전처리 이미지 선택
    
    # find objects on a frame by using YOLO
    results = tfnet.return_predict(screen[:-130, :, :]) # 하단 자르기
    # create a list of detected traffic lights (might be several on a frame)
    traffic_lights = [] # 신호등 목록
    color_detected = None # 인식된 신호등 색
    distance = 1 # 기본 거리 (무한)

    for color, color2, result in zip(colors, colors2, results):
        top_left = (result['topleft']['x'], result['topleft']['y'])
        bottom_right = (result['bottomright']['x'], result['bottomright']['y'])
        label = result['label']
        confidence = result['confidence']
        text = '{}: {:.0f}%'.format(label, confidence * 100)

        
         # 신호등 처리
        if label == 'traffic light' and confidence > 0.3:
            if 220 <= result['topleft']['x'] <= 630:
                traffic_lights.append(result)

            color = color2
            screen = cv2.rectangle(screen, top_left, bottom_right, color, 6)
            screen = cv2.putText(screen, text, top_left, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)


        # 차량 처리
        if label == 'car' or label == 'bus' or label == 'truck' or label == 'train': #if label in ['car', 'bus', 'truck', 'train']: 이렇게도 가능한 코드
            screen, car_distance = distance_to_car(screen, top_left, bottom_right)

            if car_distance and 0 <= car_distance < distance:
                distance = car_distance

            screen = cv2.rectangle(screen, top_left, bottom_right, color, 6)
            screen = cv2.putText(screen, text, top_left, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        
        # 사람 처리
        if label == 'person':
            screen, person_distance = distance_to_human(screen, top_left, bottom_right)

            if person_distance and 0 <= person_distance < distance:
                distance = person_distance

            screen = cv2.rectangle(screen, top_left, bottom_right, color, 6)
            screen = cv2.putText(screen, text, top_left, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    
     # 신호등 있으면 색 인식
    if traffic_lights:
        screen, color_detected = light_recog(screen, direct, traffic_lights)

    return screen, color_detected, distance


# 디버깅용 실행 루프
def main():
    #global tfnet #바깥의 yolo_detection함수에서도 접근가능하게
   
    
    while True:
        #screen = grab_screen()
        #screen = capture_screen()  # 이거는 모니터 전체 캡쳐라고 chatgpt가 말하지만 워낙 구라쟁이라서 믿을수가
        screen, color_detected, obj_distance = yolo_detection(screen, 0)

        if color_detected:
            print("Color detected: " + color_detected)
        if obj_distance != 1:
            print("Distance to obstacle: {}".format(obj_distance))

        cv2.imshow("Frame", screen)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()

#distance_to_car()	차량 충돌 거리 계산 및 경고 표시
#yolo_detection()	객체별 처리 (신호등, 차량, 보행자)
#main()	실시간 YOLO 시각화 디버깅 루프


########여기까지 #2