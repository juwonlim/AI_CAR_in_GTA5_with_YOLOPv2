#파일	연동 관계
#data_collect.py	원본 h5 생성자 (→ 이 파일에서 source_path 가져옴)
#data_balancing.py	data_collect.py가 만들어낸  원본 h5를 읽고 균형 잡힌 학습 데이터셋 생성
#train.py	training_data_balanced.h5를 불러와 모델 학습
#drive.py	model-xxx.h5 (훈련된 모델) 로딩해서 AI 주행 수행
#data_balancing.py는 네가 직접 플레이할 때는 아무 역할을 하지 않지만, 그때 저장된 데이터를 "훈련하기 좋게 정리"해주는 중요한 전처리 도구




import sys
import os
# 현재 스크립트 기준으로 상위 폴더(AI_GTA5 루트)를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #이건 현재 파일 기준으로 상위 폴더(=루트) 를 자동으로 찾아주는 방식


import h5py  # HDF5 파일을 읽고 쓰기 위한 라이브러리

from data_collection.data_collect import path as source_path  # 원본 HDF5 데이터 경로를 import


# 데이터를 저장할 대상 경로 (.h5 형식)
#dest_path = "F:\Graduation_Project\\training_data_balanced.h5" 작성자 원본
dest_path = "training/training_data_balanced.h5" #내가 수정한 경로


# 대상 HDF5 파일 생성 및 3개의 데이터셋 생성 (이미지, 조향값, 주행 메트릭)
destination = h5py.File(dest_path, 'w')
destination.create_dataset('img', (0, 240, 320, 3), dtype='u1', maxshape=(None, 240, 320, 3), chunks=(30, 240, 320, 3))
destination.create_dataset('controls', (0, 2), dtype='i1', maxshape=(None, 2), chunks=(30, 2))
destination.create_dataset('metrics', (0, 2), dtype='u1', maxshape=(None, 2), chunks=(30, 2))



# 데이터를 대상 HDF5 파일에 저장하는 함수
def save(data_img, controls, metrics):
    if data_img:  # if the list is not empty , # 리스트가 비어있지 않다면
        destination["img"].resize((destination["img"].shape[0] + len(data_img)), axis=0)
        destination["img"][-len(data_img):] = data_img
        destination["controls"].resize((destination["controls"].shape[0] + len(controls)), axis=0)
        destination["controls"][-len(controls):] = controls
        destination["metrics"].resize((destination["metrics"].shape[0] + len(metrics)), axis=0)
        destination["metrics"][-len(metrics):] = metrics


# 메인 함수: 원본 데이터를 읽고 특정 조건에 따라 선별하여 저장
def main():
    source = h5py.File(source_path, 'r') # 원본 데이터 로드
    images = []
    controls = []
    metrics = []

    tuples = 0  # 저장한 데이터 튜플 수
    straights = 0 # 직진 주행 프레임 수

    for i in range(source['img'].shape[0]):
        # if speed is not 0 and not arrived at the destination
        # 속도가 0이 아니고, 목적지에 도착하지 않은 경우만 처리
        if source['metrics'][i][0] != 0 and source['metrics'][i][1] != 6:
            # save only each 5th straight drive frame
            # 조향값이 0이면 직진 → 5장마다 1장만 선택
            if source['controls'][i][1] == 0:
                add = (straights % 5 == 0)
                straights += 1
            # save all turns
            else:
                add = True  # 회전 중일 땐 전부 저장

            if add:
                images.append(source['img'][i])
                controls.append(source['controls'][i])
                metrics.append(source['metrics'][i])
                tuples += 1

                 # 10,000개씩 묶어서 저장 (약 2.5GB 분량)
                if tuples % 10000 == 0:  # every 2.5 GB
                    print(tuples)
                    save(images, controls, metrics)
                    images = []
                    controls = []
                    metrics = []

    # 남은 데이터 최종 저장
    save(images, controls, metrics)
    print("Copied: {:d} tuples from the source file".format(tuples))

    # 파일 닫기
    source.close()
    destination.close()


if __name__ == '__main__':
    main()
