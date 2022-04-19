import numpy as np
import cv2


def perspective_warp(img,
                     dst_size=(1280, 720),
                     src=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)]),
                     dst=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])):
    # Calculate image size
    img_size = np.float32([(img.shape[1], img.shape[0])])
    # Get source points of the image
    src = src * img_size
    # Adjust the destination points
    dst = dst * np.float32(dst_size)
    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    return cv2.warpPerspective(img, M, dst_size)


def inv_perspective_warp(img,
                     dst_size=(1280, 720),
                     src=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)]),
                     dst=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])):
    # Calculate image size
    img_size = np.float32([(img.shape[1], img.shape[0])])
    # Get source points of the image
    src = src * img_size
    # Adjust the destination points
    dst = dst * np.float32(dst_size)
    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    return cv2.warpPerspective(img, M, dst_size)
