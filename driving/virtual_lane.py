

import cv2

""" 
def draw_virtual_centerline(image, steering_offset, image_width=1280):
    if steering_offset is None:
        return image
    lane_center = int(image_width / 2 + steering_offset)
    cv2.line(image, (lane_center, 0), (lane_center, image.shape[0]), (0, 255, 255), 2)
    return image
"""


def draw_virtual_centerline(image, steering_offset, debug=False):
    if steering_offset is None or not debug:
        return image

    # BEV용 투시변환 좌표 (720p 예시)
    src = np.float32([[550, 450], [730, 450], [0, 720], [1280, 720]])
    dst = np.float32([[500, 0], [780, 0], [500, 720], [780, 720]])
    M = cv2.getPerspectiveTransform(src, dst)

    bev = cv2.warpPerspective(image, M, (1280, 720))
    cv2.line(bev, (640, 720), (640, 0), (0, 255, 0), 2)

    Minv = cv2.getPerspectiveTransform(dst, src)
    output = cv2.warpPerspective(bev, Minv, (1280, 720))
    return output
