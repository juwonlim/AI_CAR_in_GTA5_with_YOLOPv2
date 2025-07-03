"""
Histogram of turns (for future balancing of data)
"""

"""
histogram.py 파일은
수집된 조향값(steering angle)의 분포를 히스토그램으로 시각화해서
데이터의 편향 정도(직진이 많은지, 회전이 고르게 분포되는지 등) 를 확인하는 도구


사용용도 : 데이터 수집 후 확인용, 너무 0(직진) 값이 몰려 있다면 → data_balancing.py 같은 걸로 필터링해서 직진 비율 줄이고, 회전 데이터 보완 가능



필요 전제 : data_collect.py에서 만들어지는 .h5 파일은 controls[:, 1]에 **조향값(-1, 0, 1)**을 담고 있어야 함

          즉, [-1, 0, 1]만 있어도 되고, 확장하면 [-5 ~ 5] 같은 finer-grain 조향값도 처리 가능
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #이건 현재 파일 기준으로 상위 폴더(=루트) 를 자동으로 찾아주는 방식이야


import h5py  # HDF5 파일 입출력 라이브러리
import matplotlib.pyplot as plt # 히스토그램 시각화용
import numpy as np

from data_collection.data_collect import path  # 저장된 원본 H5 데이터 경로 불러오기

n_bins = [x - 0.5 for x in range(-10, 12)] # 히스토그램의 bin 경계값 설정: -10부터 +11까지 1단위로 쪼갠 bin

data = h5py.File(path, 'r') # HDF5 데이터 파일 열기



# 조향값(controls[:, 1])을 모두 읽어서 히스토그램으로 그리기
fig, axs = plt.subplots()
axs.hist([d[1] for d in data['controls'][:]], bins=n_bins)

data.close()


# X축 눈금: -10 ~ 10까지
plt.xticks(np.arange(-10, 11, step=1))
plt.show()
