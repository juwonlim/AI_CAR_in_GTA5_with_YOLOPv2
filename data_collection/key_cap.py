
# Citation: Box Of Hats (https://github.com/Box-Of-Hats)

"""
Module for reading keys from a keyboard
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #이건 현재 파일 기준으로 상위 폴더(=루트) 를 자동으로 찾아주는 방식이야

import win32api as wapi

# 키 입력을 감지할 키 목록 초기화
keyList = ["\b"] # 백스페이스 포함
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789,.'£$/\\":
    keyList.append(char)

 
def key_check(): # 현재 눌린 키보드 키들을 검사해서 리스트로 반환
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys


#gamepad_cap.py (수정됨)	키보드로 throttle/steering/R/L 인식
#key_cap.py	일시정지 중 Y 또는 N 눌렀는지 확인
#data_collect.py	위 두 파일을 이용해서 데이터 수집, 저장 여부 결정

"""

이 key_cap.py는 간단하지만 중요한 파일이야.
키보드에서 어떤 키가 눌렸는지를 체크해서 리스트로 반환해주는 유틸리티 모듈



네 질문: 이 파일이 "키보드로 조작하는 너에게 필요한가?"
필요해.

왜냐하면,
data_collect.py에서 RB 누르면 일시정지되고
그때 “Y/N” 입력을 받아서 최근 수집 데이터 저장 여부를 결정하잖아?

그때 사용되는 키 체크가 바로 이 key_check() 함수

"""

"""
Module for reading keys from a keyboard

- 현재 눌려 있는 키보드 키들을 리스트로 반환해주는 모듈
- 예: ['W', 'A'] 또는 ['Y'], ['N'] 같은 식으로 결과 나옴
"""