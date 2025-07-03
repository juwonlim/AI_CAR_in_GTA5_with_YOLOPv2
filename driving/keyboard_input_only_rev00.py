# keyboard_input_only_rev00.py (pygame 제거 버전)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #이걸로 상위폴더 인식, data_collection에 있는 key_cap.py파일 인식

from data_collection.key_cap import key_check

class AutonomousControl:
    def __init__(self):
        self.auto_drive = False

    def check_key_events(self):
        """
        키보드 이벤트를 감지하여 자율주행 상태를 업데이트
        """
        keys = key_check()
        if 'Y' in keys:
            self.auto_drive = True
            print("[KEYBOARD] 자율주행 ON (Y키 입력)")
        elif 'N' in keys:
            self.auto_drive = False
            print("[KEYBOARD] 자율주행 OFF (N키 입력)")

    def is_auto_drive_enabled(self):
        """
        자율주행 상태 반환
        """
        return self.auto_drive

    def is_exit_pressed(self): 
        keys = key_check()
        return 'ESC' in keys


# 외부에서 쓸 수 있도록 인스턴스 생성
controller = AutonomousControl()
