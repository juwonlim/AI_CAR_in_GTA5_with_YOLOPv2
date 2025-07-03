import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #이건 현재 파일 기준으로 상위 폴더(=루트) 를 자동으로 찾아주는 방식이야

from enum import Enum

# 주행 방향을 표현하는 열거형 클래스 (enum)
class Direct(Enum):
    STRAIGHT = 0  # 직진
    LEFT = 1   # 좌회전
    RIGHT = 2 # 우회전
    SLIGHTLY_LEFT = 3 # 약간 좌회전
    SLIGHTLY_RIGHT = 4 # 약간 우회전
    U_TURN = 5 # 유턴
    ARRIVED = 6  # 목적지 도착


"""
이 파일의 쓰임새는?
img_process.py에서 방향 화살표 인식 (→ int값으로 추출)

그 int를 이 Direct enum에 매핑해서 의미를 부여함

예: direct == Direct.ARRIVED.value → 도착 여부 판단 (drive.py에서 사용됨)
                                               drive.py 등에서 목적지 도착 판단에 사용 중


"""