#차선, 앞차 인식 데이터 수집까지 완성된 버전
#차간거리 유지는 아직임

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 상위 폴더 접근 가능하게

import threading #여러작업동시처리, 즉 ,주행 끊김없이 데이터 저장 목적
import time #프로그램 속도를 제어하거나, 시간 경과를 측정할 때 , ex) sleep(0.015), time()
import winsound #속도 초과 경고 같은 걸 소리로 알려주기 위해
import h5py  # HDF5 파일 입출력용
import tensorflow as tf
import numpy as np

from data_collection.navigation_img_process import img_process  # GTA5 화면 캡처 및 처리 (네비게이션 전용)



### [추가] YOLO 객체 인식 모듈
from object_detection.object_detect import yolo_detection
from data_collection.gamepad_cap import Gamepad  # 게임패드/키보드 입력 감지
from data_collection.key_cap import key_check  # 키보드 입력 감지

# [추가] 차선인식 모듈
from data_collection.preprocess import get_preprocessed
from object_detection.lane_detect import hough_lines, construct_lane
from object_detection.lane_detect import visualize_lane #gta5칼라 이미지에 차선을 인식시키는 것

### [추가] TensorFlow GPU 메모리 4GB 제한 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # 4GB까지만 할당
            )
    except RuntimeError as e:
        print(e)

lock = threading.Lock()  # 저장 중 데이터 충돌 방지용

# 저장할 파일 경로
path = "training/training_data_by_user_drive.h5"
os.makedirs(os.path.dirname(path), exist_ok=True)

# HDF5 파일 열기
data_file = None
if os.path.isfile(path):
    data_file = h5py.File(path, 'a')  # 있으면 이어쓰기
else:
    data_file = h5py.File(path, 'w')  # 없으면 새로 생성
    # 기본 데이터셋 생성
    data_file.create_dataset('img', (0, 180, 320, 3), dtype='u1', maxshape=(None, 180, 320, 3), chunks=(30, 180, 320, 3))
    data_file.create_dataset('controls', (0, 2), dtype='i1', maxshape=(None, 2), chunks=(30, 2))
    data_file.create_dataset('metrics', (0, 2), dtype='u1', maxshape=(None, 2), chunks=(30, 2))
    ### [추가]
    data_file.create_dataset('lanes', (0, 4), dtype='i2', maxshape=(None, 4), chunks=(30, 4))  # 왼쪽/오른쪽 차선 좌표
    data_file.create_dataset('obj_distance', (0, 1), dtype='f2', maxshape=(None, 1), chunks=(30, 1))  # 앞차 거리




# 데이터를 저장하는 함수
#여기서 data_img(--> training_img)가 비어있으면, lanes,controls 등 아무것도 저장되지 않음
#그래서 training_img.append(screen)이 중요
#training_img가 비어 있으면 저장 조건 자체가 무효
#반면 training_img에 최소 1장이라도 이미지가 들어가 있으면:
#그 시점의 lanes, controls, metrics, obj_distance 전부가 HDF5로 같이 저장됨
#왜냐면 save()는 이 5개 리스트를 한꺼번에 .resize() + .append() 하는 구조
def save(data_img, controls, metrics, lanes, obj_distances):
    with lock:
        # 각 데이터셋 존재 여부 확인 후 없으면 생성
        if 'img' not in data_file:
            data_file.create_dataset('img', (0, 180, 320, 3), maxshape=(None, 180, 320, 3),
                                     dtype='u1', chunks=(30, 180, 320, 3))
        if 'controls' not in data_file:
            data_file.create_dataset('controls', (0, 2), maxshape=(None, 2),
                                     dtype='f', chunks=(30, 2))
        if 'metrics' not in data_file:
            data_file.create_dataset('metrics', (0, 2), maxshape=(None, 2),
                                     dtype='f', chunks=(30, 2))
        if 'lanes' not in data_file:
            data_file.create_dataset('lanes', (0, 4), maxshape=(None, 4),
                                     dtype='i', chunks=(30, 4))
        if 'obj_distance' not in data_file:
            data_file.create_dataset('obj_distance', (0,), maxshape=(None,),
                                     dtype='f', chunks=(30,))

        if data_img: #h5 저장 조건!!!!
            data_file["img"].resize((data_file["img"].shape[0] + len(data_img)), axis=0)
            data_file["img"][-len(data_img):] = data_img
            data_file["controls"].resize((data_file["controls"].shape[0] + len(controls)), axis=0)
            data_file["controls"][-len(controls):] = controls
            data_file["metrics"].resize((data_file["metrics"].shape[0] + len(metrics)), axis=0)
            data_file["metrics"][-len(metrics):] = metrics
            data_file["lanes"].resize((data_file["lanes"].shape[0] + len(lanes)), axis=0)
            data_file["lanes"][-len(lanes):] = lanes
            data_file["obj_distance"].resize((data_file["obj_distance"].shape[0] + len(obj_distances)), axis=0)
            data_file["obj_distance"][-len(obj_distances):] = obj_distances

# 최근 데이터 삭제 함수
#P로 일시정지하고 → N 누르면 최근 최대 500프레임(약 15초) 분량 데이터를 삭제 --> 잘못운행한 구간 삭제하는 기능
def delete(session):
    frames = session if session < 500 else 500
    data_file["img"].resize((data_file["img"].shape[0] - frames), axis=0)
    data_file["controls"].resize((data_file["controls"].shape[0] - frames), axis=0)
    data_file["metrics"].resize((data_file["metrics"].shape[0] - frames), axis=0)
    data_file["lanes"].resize((data_file["lanes"].shape[0] - frames), axis=0)
    data_file["obj_distance"].resize((data_file["obj_distance"].shape[0] - frames), axis=0)

# 메인 루프
def main():
    debug_view = None
    gamepad = Gamepad()
    gamepad.open()

    alert_time = time.time()
    close = False
    pause = True
    session = 0
    training_img = []
    controls = []
    metrics = []
    lanes = []
    obj_distances = []

    print("Press RB on your gamepad or keyboard 'K' to start recording")
    print("프로그램 시작됨. K를 눌러 녹화를 시작하세요.")

    while not close:
        while not pause:
           

            throttle, steering = gamepad.get_state()  # 게임패드로부터 throttle, steering 읽기
            ignore, screen, speed, direction = img_process("Grand Theft Auto V")  # 화면 캡처 및 차량 속도
            
        
            

            ### [추가] YOLO로 앞차 감지
            _, _, obj_distance = yolo_detection(screen, direct=0)
            if obj_distance is None:
                obj_distance = 1.0  # 기본값: 앞차 없음
            obj_distances.append([obj_distance])

           #이거 아주 중요함 def save함수위에 설명 볼것
            training_img.append(screen) # 차선인식을 저장, 이 한줄 추가해야 lane값이 h5파일에 저장된다는 !!!!!!
            
            controls.append([throttle, steering]) #스로틀과 스티어링 저장
            metrics.append([speed, direction]) #속도와 방향 저장
            session += 1

            # 속도 60km/h 초과시 경고음
            if speed > 60 and time.time() - alert_time > 1:
                winsound.PlaySound('.\\resources\\alert.wav', winsound.SND_ASYNC)
                alert_time = time.time()

            # 30프레임마다 비동기로 저장
            if len(training_img) % 30 == 0:
                threading.Thread(target=save, args=(training_img, controls, metrics, lanes, obj_distances)).start()
                training_img = [] #이게 비어있으면 아래의 lanes만 누적되고 if training_img:조건에 의해 save()함수 내부가 동작안함
                controls = []
                metrics = []
                lanes = []
                obj_distances = []

            time.sleep(0.015)  # CPU 부하 줄이기

            # 일시정지/저장/삭제 핸들링
            if gamepad.get_RB() or 'P' in key_check():
                pause = True
                print('Paused. Save the last 15 seconds?')

                keys = key_check()
                while ('Y' not in keys) and ('N' not in keys):
                    keys = key_check()

                if 'N' in keys:
                    delete(session)
                    training_img = []
                    controls = []
                    metrics = []
                    lanes = []
                    obj_distances = []
                    print('Deleted.')
                else:
                    print('Saved.')

                print('To exit the program press LB or keyboard L.')
                session = 0
                time.sleep(0.5)

            # ROI 추출
            #여기서 부터 차선인식해서 저장
            #roi = get_preprocessed() #이렇게 했더니 TypeError: only integers, slices (`:`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices 이런 에러 발생
            roi, original_img = get_preprocessed()
            
            if original_img is None or not isinstance(original_img, np.ndarray):
                print("[ERROR] original_img is not valid.")
                return  # 또는 continue
                       
            
            cropped_roi = original_img[200:550, :, :]



            if roi is not None:
                #cropped_roi = roi[200:550, :, :] #IndexError: too many indices for array ,roi는 아마도 2차원 배열 (예: (height, width))인데, 3차원 배열처럼 [200:550, :, :] 슬라이싱을 했기 때문에 발생한 에러
                cropped_roi = roi[200:550, :] #roi가 2차원 (HEIGHT,WIDTH 2차원배열, 흑백)일 경우 이렇게 2차원으로 접근, ROI가 3차원이면 주석된 윗줄이 맞음
                lines = hough_lines(cropped_roi)
                lane_result = construct_lane(lines)

                 # 차선 좌표 검증
                if not lane_result["lanes"]:
                    lanes.append([0, 0, 0, 0])
                    continue  #return은 main() 전체 종료, continue는 다음 루프로 넘어감
               
                
                left_lane = lane_result["lanes"][0] if lane_result["lanes"][0] else [0, 0, 0, 0]
                right_lane = lane_result["lanes"][1] if lane_result["lanes"][1] else [0, 0, 0, 0]
                lanes.append([left_lane[0], left_lane[2], right_lane[0], right_lane[2]])
                
                 #차선 시각화 함수 호출은 여기로
                if lane_result and lane_result["lanes"] and lane_result["stop_line"]: #이 조건 없이 무조건 호출하면 blended가 None이 될 수 있고, resize 시도하면서 프로그램이 down
                    visualize_lane(lane_result, original_img)

                #visualize_lane(lane_result, original_img) #lane_result["lanes"] 유효성 검증 후에 호출해야 함
            else:
                lanes.append([0, 0, 0, 0])  # 실패했을 경우 기본값

               

  

        # 녹화 재개
        keys = key_check()
        #print(f"[DEBUG] Key input: {keys}")  # <- 추가 , 입력값 출력하는 코드,  디버깅 용도, 너무 디버그 줄이 계속 나와서 주석처리
        if 'K' in keys:
            print(f"[DEBUG] Key input: {keys}")
        if gamepad.get_RB() or 'K' in keys:
            pause = False
            print('Unpaused by keyboard or gamepad')
            time.sleep(1)

        # 정확히 'L'만 눌렸을 때만 종료
        elif gamepad.get_LB() or ('L' in keys and len(keys) == 1): # 이 elif문 아래에 5개의 코드줄이 없으면 아래에 if --name__문에서 에러발생
                                                                   #이 조건이 빈 코드 블럭(아무 동작도 없음)이라, 이후의 실제 종료처리 코드가 주석 처리되어 있어도 main() 루프가 빠져나와 종료될 수 있음
                                                                   # close = True 설정이 없으면 while 루프는 계속 돌아야 하는데, 이 부분도 빠져 있어 예상치 못한 종료가 일어나는 겁니다
               print("Saving data and closing the program.")
               save(training_img, controls, metrics, lanes, obj_distances) #이미지,차선,거리,조작 정보 저장
               data_file.close()
               gamepad.close()
               close = True
               

        
     
        



# 프로그램 시작
if __name__ == '__main__':
    print("data_collect.py 실행됨.")
    main()
