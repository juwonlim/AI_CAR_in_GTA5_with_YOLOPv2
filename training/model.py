"""
NN model
"""
# Keras에서 사용할 레이어, 모델 정의 관련 함수들 import
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten, Concatenate, Input, MaxPooling2D
from keras.models import Model

# 이미지와 레이더 입력 크기를 정의한 상수 import
from training.utils import INPUT_SHAPE, RADAR_SHAPE


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #이건 현재 파일 기준으로 상위 폴더(=루트) 를 자동으로 찾아주는 방식이야 




# original Nvidia model
# def build_model(args):
#     """
#     NVIDIA model used
#     Image normalization to avoid saturation and make gradients work better.
#     Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
#     Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
#     Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
#     Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
#     Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
#     Drop out (0.5)
#     Fully connected: neurons: 100, activation: ELU
#     Fully connected: neurons: 50, activation: ELU
#     Fully connected: neurons: 10, activation: ELU
#     Fully connected: neurons: 1 (output)
#     # the convolution layers are meant to handle feature engineering
#     the fully connected layer for predicting the steering angle.
#     dropout avoids overfitting
#     ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
#     """
#     model = Sequential()
#     model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
#     model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
#     model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
#     model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
#     model.add(Conv2D(64, (3, 3), activation='elu'))
#     model.add(Conv2D(64, (3, 3), activation='elu'))
#     model.add(Dropout(args.keep_prob))
#     model.add(Flatten())
#     model.add(Dense(100, activation='elu'))
#     model.add(Dense(50, activation='elu'))
#     model.add(Dense(10, activation='elu'))
#     model.add(Dense(1))
#     model.summary()
#
#     return model





# original + radar and speed info added
def build_model(args):
      # ░░░░ 이미지 처리 파이프라인 (camera image) ░░░
    # image model
    img_input = Input(shape=INPUT_SHAPE)   # 이미지 입력 정의 (예: 320x240x3)
    img_model = (Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))(img_input)  # 픽셀 정규화 (-1.0 ~ 1.0 범위로)
    
    # 5x5 필터, 24채널, stride 2 → 다운샘플링 + 특징 추출
    img_model = (Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))(img_model)
    img_model = (Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))(img_model)
    img_model = (Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))(img_model)
    
     # 3x3 필터로 더 미세한 특징 추출
    img_model = (Conv2D(64, (3, 3), activation='elu'))(img_model)
    img_model = (Conv2D(64, (3, 3), activation='elu'))(img_model)




    img_model = (Dropout(args.keep_prob))(img_model)  # 과적합 방지를 위한 Dropout 적용
    img_model = (Flatten())(img_model)  # Fully Connected 레이어에 연결하기 위해 평탄화
    img_model = (Dense(100, activation='elu'))(img_model) # 전결합층 → 특징 조합, 최종 예측값 준비
 
    # radar model
    # 레이더 이미지 처리 파이프라인
    radar_input = Input(shape=RADAR_SHAPE)  # 레이더 입력 정의 (예: 20x20x1 또는 20x20x3)
    
    # 32채널 conv 레이어 → 공간 특징 추출
    radar_model = (Conv2D(32, (5, 5), activation='elu'))(radar_input)
    radar_model = (MaxPooling2D((2, 2), strides=(2, 2)))(radar_model)
    
    # 더 깊은 특징 추출
    radar_model = (Conv2D(64, (5, 5), activation='elu'))(radar_model)
    radar_model = (MaxPooling2D((2, 2), strides=(2, 2)))(radar_model)
    
    # 레이더도 과적합 방지용 Dropout 적용
    radar_model = (Dropout(args.keep_prob / 2))(radar_model)
    
    # Flatten + Dense → 레이더 특징 벡터화
    radar_model = (Flatten())(radar_model)
    radar_model = (Dense(10, activation='elu'))(radar_model)

    # speed
    # 속도 입력 (1차원 스칼라 입력)
    speed_input = Input(shape=(1,)) # 단일 float형 입력 (현재 속도)

    # combined model
    #최종 결합 
    out = Concatenate()([img_model, radar_model])  # 이미지 + 레이더 출력 연결
    out = (Dense(50, activation='elu'))(out) # 전결합층 통과
    out = Concatenate()([out, speed_input]) # 속도 정보 추가 결합
    
    # 최종 출력 준비
    out = (Dense(10, activation='elu'))(out)
    out = (Dense(1))(out) # 최종 출력값: 조향각 (float scalar)

    # 모델 구성: 입력 3개, 출력 1개
    final_model = Model(inputs=[img_input, radar_input, speed_input], outputs=out)
    
    # 모델 구조 콘솔에 출력
    final_model.summary()

    return final_model

#여기까지 핵심구조요약
#camera image	CNN (Nvidia 구조 기반)	도로/장애물/차선 정보
#radar image	작은 CNN + maxpool	근거리 물체 인식
#speed	scalar + Dense	현재 속도 반영
#최종 출력	Dense(1)	조향각 예측 (steering angle)

#현재 args.keep_prob 값은 dropout 비율에 사용됨 (예: 0.5 → 50% dropout)
#Keras의 Functional API를 사용해 입출력이 명확하게 분리됨
#모델 학습은 train.py에서 이 build_model()을 호출해 이뤄짐


