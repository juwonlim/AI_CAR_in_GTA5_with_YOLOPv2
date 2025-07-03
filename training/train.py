# This code based on Siraj Raval's code (https://github.com/llSourcell/How_to_simulate_a_self_driving_car)

"""
#train.py 핵심 기능 요약 :
이 파일은 전체 자율주행 모델을 학습하는 코드로, 다음을 수행해:
학습 데이터를 로드하고 train/validation으로 나눔
모델을 생성하고 기존 가중치를 불러옴
지정된 epoch만큼 모델을 학습함
validation loss 기준으로 가장 좋은 모델을 저장함

#주요 부분별 주석 요약 :
1. 라이브러리 및 모듈 import
h5py, keras, sklearn 등 모델 학습에 필요한 패키지를 불러와
model.py에 정의된 모델 구조, utils.py에 있는 데이터 배치 생성기를 사용함


#전체 흐름 순서 정리:
명령줄 인자 파싱 (argparse)
데이터 로드 (load_data)
모델 정의 (build_model)
기존 가중치 불러오기 (load_weights)
학습 수행 (train_model)




#내 수준에서 기억해야 할 키포인트:

h5py.File(path, 'r') :	저장된 데이터를 로드해 (이미지, 레이더, 속도 등)
build_model(args)	: 우리가 정의한 자율주행 모델 구조를 생성해
model.fit_generator(...) :	데이터를 한 번에 다 넣지 않고 조금씩 나눠서 학습함
ModelCheckpoint(...) :	val_loss 기준으로 가장 좋은 모델만 저장 가능
args	: command line으로 설정값 전달 (예: dropout 비율, learning rate 등)



"""


"""
Training module. Based on "End to End Learning for Self-Driving Cars" research paper by Nvidia.
"""
import tensorflow as tf
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  #cpu만 사용
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #이건 현재 파일 기준으로 상위 폴더(=루트) 를 자동으로 찾아주는 방식
                                                                                #이게 제일 처음 나아야함. from ddata_collectin보다 늦게 나오면 경로인식못함

""" 
#여기부터
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # GPU 메모리 점유를 필요한 만큼만 사용
set_session(tf.Session(config=config))
#여기까지 gpu강제로 사용을 도움

#다음 3줄의 코드로 필요할 때만 메모리 할당(gpu메모리인가?)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
"""



import argparse  # 명령줄 인자를 쉽게 처리할 수 있게 해주는 모듈
import h5py  # .h5 형식의 데이터 파일을 읽고 쓰기 위한 라이브러리
import numpy as np
from keras.callbacks import ModelCheckpoint # keras 모델 학습 중 자동 저장 기능
from keras.models import load_model # 학습된 모델을 불러오는 함수
from keras.optimizers import Adam # 딥러닝에서 널리 쓰이는 최적화 알고리즘
from sklearn.model_selection import train_test_split  # to split out training and testing data  # 학습/검증 데이터 나누기

# path with training files
from data_collection.data_collect import path #data_collect.py파일에서 path를 불러옴
from training.model import build_model # 우리가 만든 딥러닝 모델 불러오기
# helper class
from training.utils import batch_generator # 배치 생성기 (utils.py에서 정의)



# for debugging, allows for reproducible (deterministic) results
np.random.seed(0) # 랜덤값이 고정되도록 설정 (재현성을 위해)


def load_data(args):
    """
    Load training data and split it into training and validation set
    학습 데이터를 h5 파일에서 불러오고, 학습/검증 세트로 나눔
    """
    data = h5py.File(path, 'r') # h5 파일 열기 (read only 모드)
    # list of all possible indexes
    indexes = list(range(data['img'].shape[0])) # 전체 이미지 인덱스를 리스트로 만듦
    
    # split the data into a training (80), testing(20), and validation set
      # 학습/검증 세트로 나누기 (기본은 80:20)
      # train_test_split을 이용해서 학습과 검증 데이터로 분할
    indexes_train, indexes_valid = train_test_split(indexes, test_size=args.test_size, random_state=0)

    return data, indexes_train, indexes_valid


""" 
#이 함수는 train.py에서 훈련이 완료된 파일을 로드하는 기능임. 나는 처음부터 새로운 모델로 학습을 할거니까 
def load_weights(model):
    
    #Load weights from previously trained model
    #이전에 학습한 모델(base_model.h5)의 가중치를 불러와서 현재 모델에 적용
    
    #prev_model = load_model("..\\training\\base_model.h5")  # 예전 git주인의 h5파일
    prev_model = load_model("..\\training\\training_data_by_user_drive.h5")
    model.set_weights(prev_model.get_weights())   # 그 모델의 가중치를 현재 모델에 복사

    return model

"""
def train_model(model, args, data, indexes_train, indexes_valid):
    """
    Train the model
    모델 학습을 수행하는 함수
    """
    # Saves the model after every epoch.
    # quantity to monitor, verbosity i.e logging mode (0 or 1),
    # if save_best_only is true the latest best model according to the quantity monitored will not be overwritten.
    # mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is
    # made based on either the maximization or the minimization of the monitored quantity. For val_acc,
    # this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically
    # inferred from the name of the monitored quantity.

    ## Epoch마다 모델을 저장하기 위한 콜백 함수 설정
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', # 저장할 파일명 (epoch마다 버전)
                                 monitor='val_loss',  # 검증 손실(val_loss)이 좋아질 때 저장 ( # 모니터할 값은 검증 손실)
                                 verbose=0, # 로그 출력 안 함
                                 save_best_only=args.save_best_only, # 가장 좋은 모델만 저장할지 여부
                                 mode='auto')   # val_loss의 경우 자동으로 'min' 인식

    # calculate the difference between expected steering angle and actual steering angle
    # square the difference
    # add up all those differences for as many data points as we have
    # divide by the number of them
    # that value is our mean squared error! this is what we want to minimize via
    # gradient descent

    # 모델을 학습할 때 사용할 손실 함수와 최적화 알고리즘 정의
    # 손실 함수: 평균제곱오차
    # 옵티마이저: Adam (학습률은 명령줄 인자로 받음)

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    # Fits the model on data generated batch-by-batch by a Python generator.

    # The generator is run in parallel to the model, for efficiency.
    # For instance, this allows you to do real-time data augmentation on images on CPU in
    # parallel to training your model on GPU.
    # so we reshape our data into their appropriate batches and train our model simultaneously
    
    # 실제 학습을 시작
     # batch_generator는 이미지를 배치 단위로 생성해서 넘겨주는 제너레이터 함수
    
    #with tf.device('/GPU:0'): #gpu사용 강제함-- 사용안함. GPU연산이 잘안되어서 
    model.fit_generator(batch_generator(data, indexes_train, args.batch_size, True), # 학습용 데이터 생성기
                        steps_per_epoch=len(indexes_train) / args.batch_size,  # epoch마다 step 수
                        epochs=args.nb_epoch,  # 전체 epoch 횟수
                        max_queue_size=1,  # 큐 크기 (병렬처리 최소화)
                        validation_data=batch_generator(data, indexes_valid, args.batch_size, False),  # 검증용 생성기
                        validation_steps=len(indexes_valid) / args.batch_size,  # 검증용 step 수
                        callbacks=[checkpoint],  # 체크포인트 저장
                        verbose=1) # 학습 로그 출력   


# for command line args
def s2b(s):
    """
    Converts a string to boolean value
    문자열을 boolean으로 변환하는 함수
    예: 'True', 'yes', '1' → True
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
     전체 학습 프로세스를 실행하는 메인 함수
    1. 명령줄 인자를 받아오고
    2. 데이터 로드 → 모델 구성 → 가중치 로드 → 모델 학습
    """
    # The argparse module makes it easy to write user-friendly command-line interfaces.
       # argparse를 사용해서 학습 관련 설정값들을 받는다
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default=path)  # 데이터 경로
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)  # 검증 비율
    parser.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=0.5) # 드롭아웃 비율
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=40)  # 학습 epoch 수,원래는 200이었음
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=500)  #배치 사이즈,원래 500, 늘릴수도있다??
    parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default='true') # 최적 모델만 저장
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)  # 학습률
    args = parser.parse_args()  # 위 인자들을 실제로 받음

    # print parameters
     # 설정값들을 보기 좋게 출력
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))  # key := value 형태로 출력
    print('-' * 30)

    # load data
    # 학습 데이터 로딩 (h5파일 로드 + 인덱스 분할)
    data = load_data(args)
    
    # build model
    # 모델 구성
    model = build_model(args)
    
    # load previous weights
     # 기존에 학습한 모델이 있다면 그 가중치를 불러옴
    #model = load_weights(model) #새로 학습을 시작할 것이라서 기존에 학습된 데이터를 불러올게 없음 (250424 pm7:00)
    
    # train model on data, it saves as model.h5
    # 학습 시작
    train_model(model, args, *data)


if __name__ == '__main__':
    main() # 스크립트를 직접 실행하면 main()이 호출됨
