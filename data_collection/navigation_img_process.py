
"""
Module for preprocessing screen captures from GTA5.
- 화면을 캡처하고
- 속도 숫자 (세자리) + 방향 화살표 인식
- 이미지 리사이즈 후 반환

원래 파일명이 img_process.py였고 이미지 전처리도 했었으나, 네비게이션용도로 변경함
내부 함수도 온전히 네비게이션 용도의 함수만 남아두게 정리함
1차 정리함(250506)
"""

import sys
import os
import numpy as np
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #이건 현재 파일 기준으로 상위 폴더(=루트) 를 자동으로 찾아주는 방식

# 프로젝트 루트 기준으로 절대 경로 생성
base_dir = os.path.dirname(os.path.abspath(__file__))  # navigation_img_process.py 경로
RESOURCE_PATH = os.path.join(os.path.dirname(__file__), 'resources')
#resource_path = os.path.join(base_dir, 'resources', 'digits.npy')



digits = np.load(os.path.join(RESOURCE_PATH, 'digits.npy'))
digits_labels = np.load(os.path.join(RESOURCE_PATH, 'digits_labels.npy'))
arrows = np.load(os.path.join(RESOURCE_PATH, 'arrows.npy'))
arrows_labels = np.load(os.path.join(RESOURCE_PATH, 'arrows_labels.npy'))

import win32gui
import win32ui
import cv2
import numpy as np
import win32con


from data_collection.preprocess import grab_screen




#initKNN() 함수	np.load()가 아니라 바로 넘겨받은 배열을 reshape해야 함
#KNN 모델 로딩 함수 (파일 경로가 아닌 배열 입력으로 수정)
def initKNN(data_array, label_array, shape):
    knn = cv2.ml.KNearest_create()
    train = data_array.reshape(-1, shape).astype(np.float32)
    knn.train(train, cv2.ml.ROW_SAMPLE, label_array)
    return knn

# 숫자와 방향 화살표 인식용 KNN 모델 초기화
#이제 정확히 메모리에 올라온 .npy 기반으로 초기화
knnDigits = initKNN(digits, digits_labels, 40)
knnArrows = initKNN(arrows, arrows_labels, 90)


""" #자꾸 에러가 나서 주석처리
# 숫자와 방향 화살표 인식용 KNN 모델 (미리 학습된 .npy 사용)
knnDigits = initKNN('..\data_collection\\resources\digits.npy',
                    '..\data_collection\\resources\digits_labels.npy', 40)
knnArrows = initKNN('..\data_collection\\resources\\arrows.npy',
                    '..\data_collection\\resources\\arrows_labels.npy', 90)
"""







# KNN 예측 호출 함수
#숫자추론 (네비게이션 전용)
def predict(img, knn):
    ret, result, neighbours, dist = knn.findNearest(img, k=1)
    return result

# 이미지 전처리: 흑백 + 이진화
#흑백변환, 이진화(thresholding) --> HSV,ROI,Canny등과 무관
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, -5)
    return thr

# 세 자리 숫자 → 실제 속도 정수로 변환
def convert_speed(num1, num2, num3):
    hundreds = 1
    tens = 1
    speed = 0

    if num3[0][0] != 10:
        hundreds = 10
        tens = 10
        speed += int(num3[0][0])
    if num2[0][0] != 10:
        speed += tens * int(num2[0][0])
        hundreds = tens * 10
    if num1[0][0] != 10:
        speed += hundreds * int(num1[0][0])

    return speed


# 전체 전처리 루틴
# preprocess포함, 28x28 크기로 리사이즈 --> 숫자인식용 전처리만 포함
def img_process(winName: str = "Grand Theft Auto V"):
    screen = grab_screen(winName)
    if screen is None:
        print("[ERROR] Screen capture failed.")
        return None, None, None, None  # 안전하게 4개 반환


    #여기서부터 영역표시 --> 여기서부터 캡쳐확인용 코드였으나, bfc_allocator(?)라는 memory문제가 발생해서 주석처리함.
    # 시각화용 디버깅 박스 추가 (속도 숫자 영역)
    #debug_screen = screen.copy() #본 screen을 건드리지 않고 복사본에 시각화
    #cv2.rectangle()로 좌표를 사각형으로 시각화
    #cv2.rectangle(debug_screen, (683, 567), (702, 575), (0, 255, 0), 2)  # 숫자 영역
    #cv2.rectangle(debug_screen, (18, 561), (28, 570), (255, 0, 0), 2)    # 방향 인식 영역
    #cv2.imshow()를 통해 실시간 확인.
    #cv2.imshow("Debug - Capture Areas", debug_screen)
    #cv2.waitKey(1)

    # GTA5 캡처 전체 영역 (1280x720)을 시각화
    #debug_screen = screen.copy()
    #cv2.rectangle(debug_screen, (0, 0), (1279, 719), (0, 255, 255), 3)  # 노란색 테두리
    
    # 확인용 창 띄우기
    #cv2.imshow("Debug - Full Capture Area", debug_screen)
    #cv2.resizeWindow('Debug - Full Capture Area', 320, 180)  # (1) 크기  줄이기
    #cv2.moveWindow('Debug - Full Capture Area', 1600, 850)     # (2) 오른쪽 하단으로 스크린 이동
    #cv2.waitKey(1)




    # Ji Hyun's computer
    # 👇 여기가 "속도 숫자 영역" 추출 좌표 (3자리 숫자)
    # 해상도 1280x720 기준이고, GTA5가 모니터 좌측 상단에 위치할 경우
    numbers = preprocess(screen[567:575, 683:702, :])
    # Rustam's computer
    # numbers = preprocess(screen[573:581, 683:702, :])

    # three fields for numbers
    # 숫자 세자리 분리 예측
    num1 = predict(numbers[:, :5].reshape(-1, 40).astype(np.float32), knnDigits)
    num2 = predict(numbers[:, 7:12].reshape(-1, 40).astype(np.float32), knnDigits)
    num3 = predict(numbers[:, -5:].reshape(-1, 40).astype(np.float32), knnDigits)

    # one field for direction arrows
    # Ji Hyun's computer
      # 👇 방향 화살표 인식 영역 (좌하단 또는 센터 왼쪽)
    #direct = preprocess(screen[561:570, 18:28, :]).reshape(-1, 90).astype(np.float32)  #screen[567:575, 683:702, :] ← 이건 이미지에서의 좌표 슬라이싱이 맞고, 이 좌표가 GTA5 화면 상에서 속도계 숫자 영역에 정확히 대응하는지는, 실제 해보기 전까진 확실하게 모른다.


    # Rustam's computer
    # direct = preprocess(screen[567:576, 18:28, :]).reshape(-1, 90).astype(np.float32)
    #direct = int(predict(direct, knnArrows)[0][0])

    speed = convert_speed(num1, num2, num3)
    resized = cv2.resize(screen, (320, 180)) # 학습용 이미지 크기로 변환, 16:9화면 크기, 그런데 실제로 쓰이지는 않고 data_collect.py에서 320x180으로 리사이즈함
   

    return screen, resized, speed, direct


"""
왜 해보기 전까지 알 수 없냐?
이건 실제로 운영체제의 창 위치, GTA5의 HUD 위치, 게임 해상도,
그리고 모니터에서 화면이 정확히 어디에 떠 있는지에 따라 픽셀 좌표가 바뀔 수 있기 때문이야.

특히 GTA5는 창 모드라도 Windows의 창 테두리, 타이틀바(32px), 그림자 등이 추가되기 때문에,
좌표는 항상 1:1 정해진 게 아냐.

그래서 가장 현실적인 방법은?
캡처된 이미지에서 숫자가 제대로 잡히는지 직접 시각화해서 확인하는 거야.

 예시 코드 (테스트용):
python
복사
편집
screen, resized, speed, direction = img_process()

# 속도 숫자 영역만 보기
cv2.imshow("Speed Area", screen[567:575, 683:702])
cv2.imshow("Full Frame", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
→ 이렇게 하면 그 좌표에 실제 숫자가 있는지 육안으로 바로 확인 가능해.
만약 검은 화면이거나 숫자 아닌 부분이 뜬다면 → 좌표 수정 필요.

좌표 튜닝 팁
OpenCV에서 캡처한 전체 이미지 사이즈가 1280x720과 같다고 해도
게임창 내부 콘텐츠가 그보다 작을 수 있어.

따라서 offset 조정은 ±10~30px 단위로 수동 미세조정이 필요해.

결론:

저 좌표가 실제 속도계에 대응돼? 직접 해보기 전까지 절대 모름
어떻게 확인해?  시각화해서 숫자가 잘리는지 여부로 확인
맞지 않으면? 좌표 screen[y1:y2, x1:x2] 범위를 조정하면 됨
"""

