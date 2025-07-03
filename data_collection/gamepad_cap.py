"""
Module for simulating gamepad input using keyboard keys
게임패드 없이 키보드로 throttle(W), steering(A/D), 시작/종료(R/L) 조작을 지원하는 모듈
"""

import keyboard  # 키보드 입력 감지 라이브러리

class Gamepad:
    def open(self):
        # 실제 게임패드는 열 필요 없으므로 빈 메서드 처리
        pass

    def close(self):
        # 리소스 해제 필요 없음
        pass

    def get_state(self):
        """
        현재 키보드 입력 상태를 바탕으로
        throttle(W), steering(A/D)를 반환
        """
        # W키를 누르고 있으면 throttle = 1, 아니면 0
        throttle = 1 if keyboard.is_pressed('w') else 0

        # A = 좌회전(-1), D = 우회전(+1), 아무 것도 없으면 0
        if keyboard.is_pressed('a'):
            steering = -1
        elif keyboard.is_pressed('d'):
            steering = 1
        else:
            steering = 0

        return throttle, steering

    def get_RB(self):
        """
        게임패드의 RB 버튼 대신 'r' 키로 대체
        (데이터 수집 시작/일시정지용)
        """
        return keyboard.is_pressed('r')

    def get_LB(self):
        """
        게임패드의 LB 버튼 대신 'l' 키로 대체
        (종료용)
        """
        return keyboard.is_pressed('l')
