

import cv2

def draw_virtual_centerline(image, steering_offset, image_width=1280):
    if steering_offset is None:
        return image
    lane_center = int(image_width / 2 + steering_offset)
    cv2.line(image, (lane_center, 0), (lane_center, image.shape[0]), (0, 255, 255), 2)
    return image
