# This code based on Siraj Raval's code (https://github.com/llSourcell/How_to_simulate_a_self_driving_car)

#이미지 증강, 배치처리의 핵심 함수들

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #이건 현재 파일 기준으로 상위 폴더(=루트) 를 자동으로 찾아주는 방식이야



import math

import cv2
import numpy as np
import tensorflow as tf




# 이미지와 레이더 데이터를 학습에 맞게 정형화한 입력 차원 정의
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3 #data_collect.py에서 이미지 사이즈를 무엇으로 하던간에 어차피 66x200사이즈로 전처리됨
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
RADAR_HEIGHT, RADAR_WIDTH, RADAR_CHANNELS = 20, 20, 1
RADAR_SHAPE = (RADAR_HEIGHT, RADAR_WIDTH, RADAR_CHANNELS)


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    이미지에서 하늘(상단)과 자동차 보닛(하단)을 잘라내는 함수
    → 네트워크가 주행에 불필요한 정보에 주의 주지 않도록 하기 위함
    """
    return image[90:-50, :, :]


def resize(image):
    """
    Resize the image to the input shape used by the network model
    잘라낸 이미지를 네트워크 입력 사이즈(200x66)로 리사이즈
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
     RGB 이미지를 YUV 색공간으로 변환
    (NVIDIA 자율주행 논문 모델이 YUV를 사용했기 때문)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
      crop → resize → YUV 변환 순으로 이미지 전처리 수행
    """
    image = crop(image) # 위쪽 하늘, 아래쪽 차량 보닛 제거
    image = resize(image)  # (320x180 → 200x66) 으로 줄임
    image = rgb2yuv(image)  # YUV로 변환 (Nvidia 모델 기준)
    return image


def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image vertically and horizontally (translation).
    이미지 평행이동 (translation) 을 랜덤하게 적용하는 함수.
    steering_angle(조향각)도 평행이동 정도에 따라 약간 보정해줌.

    """
    trans_x = range_x * (np.random.rand() - 0.5) # 좌우 이동 범위
    trans_y = range_y * (np.random.rand() - 0.5)  # 상하 이동 범위

    # adjusting steering angle
      # 이동량에 따른 조향각 보정
    t_x = trans_x / 25
    if t_x > 0:
        t_x = math.ceil(t_x)
        if t_x > 2:
            steering_angle += (t_x - 2)
            if steering_angle > 10:
                steering_angle = 10
    else:
        t_x = math.floor(t_x)
        if t_x < -2:
            steering_angle += (t_x + 2)
            if steering_angle < -10:
                steering_angle = -10
    

    # 실제 이미지에 평행이동(아핀 변환) 적용
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    # apply an affine transformation to an image
    image = cv2.warpAffine(image, trans_m, (width, height))  # 새로운 이미지 반환
    return image, steering_angle
##################################여기까지 #1

def random_shadow(image):
    """
    Generates and adds random shadow
    이미지에 무작위 그림자를 추가하는 함수.
    밝기와 대조에 변화를 줘서 모델이 더 다양한 상황에 적응할 수 있도록 함.
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
     # 랜덤한 두 점(x1, y1) ~ (x2, y2)로 그림자 선 정의
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]  # 전체 좌표 그리드 생성

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
     # 선 기준으로 위/아래 나눠서 그림자 영역 마스크 생성
    mask = np.zeros_like(image[:, :, 1])
    mask[np.where((ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0)] = 1

    # choose which side should have shadow and adjust saturation
     # 0 또는 1을 랜덤하게 선택해 한쪽 영역만 그림자 적용
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5) # 그림자 강도 비율

    # adjust Saturation in HLS(Hue, Light, Saturation)
     # HLS로 변환 후, 밝기(Lightness)만 줄이기
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)   # 다시 RGB로 변환


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    이미지의 밝기를 랜덤하게 조절하는 함수.
    다양한 조도 조건에서 학습이 되도록 도움.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)  # 밝기 비율: 0.8 ~ 1.2
    hsv[:, :, 2] = hsv[:, :, 2] * ratio # V 채널(밝기)에 적용
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment(image, steering_angle, range_x=250, range_y=20):
    """
    Generate an augmented image and adjust steering angle.
    (The steering angle is associated with the center image)
    주어진 이미지와 조향각에 대해 랜덤 augmentations를 수행하고,
    조향각도 그에 맞게 수정해주는 함수.
    """
    # image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    # image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle


def batch_generator(data, indexes, batch_size, is_training):
    """
    Generate training image give image paths and associated steering angles
    학습용 데이터를 배치 단위로 생성하는 제너레이터 함수.
    augmentation을 포함하며, model.fit_generator에 사용됨.
    """
    
    # preprocessing on the CPU
    # CPU에서 실행되도록 설정 (GPU가 모델 학습에 집중하도록 분리)
    with tf.device('/cpu:0'):
         # 빈 배치 배열 초기화
        images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
        radars = np.empty([batch_size, RADAR_HEIGHT, RADAR_WIDTH, RADAR_CHANNELS])
        # metrics = np.empty([batch_size, 2])
        # controls = np.empty([batch_size, 2])
        speeds = np.empty(batch_size)
        controls = np.empty(batch_size)
        while True:
            i = 0
             # 배치 1개 생성할 때까지 반복
            for index in np.random.permutation(indexes): # 인덱스 무작위 셔플
                camera = data['img'][index] # RGB 이미지 (240x320x3)
                  # radar 이미지는 특정 영역 잘라서 Blue 채널만 사용 (20x20x1)
                radar = cv2.cvtColor(camera[206:226, 25:45, :], cv2.COLOR_RGB2BGR)
                steer = data['controls'][index][1]  # 조향각

                # augmentation
                 # 학습 중이라면 augmentation 적용
                if is_training:
                    prob = np.random.rand()
                     # 조향각이 작거나, 확률 조건에 따라 augment 적용 여부 결정
                    if (abs(steer) < 0.4 and prob > 0.2) or (prob < 0.6):
                        camera, steer = augment(camera, steer)

                # add the image and steering angle to the batch
                 # 전처리 및 배치에 저장
                images[i] = preprocess(camera)
                radars[i] = radar[:, :, 2:3] # Blue 채널만 추출해서 shape 맞춤
                # controls[i] = [data['controls'][index][0] / 10, steer / 10]  # normalized throttle and steering
                controls[i] = steer / 10  # 정규화
                speeds[i] = data['metrics'][index][0] # 속도값 저장
                # metrics[i] = data['metrics'][index]
                i += 1
                if i == batch_size:
                    break
            # yield [images, metrics], controls
             # 배치 완성 → 넘겨줌
            yield [images, radars, speeds], controls
###################여기까지 #2

"""
전체 흐름 요약 :
random_shadow()	 : 무작위 그림자 생성
random_brightness() : 밝기 랜덤 조정
augment() : 평행이동 + 그림자 + 밝기 변환
batch_generator() :	모델 학습을 위한 배치 생성 (augmentation 포함)


"""