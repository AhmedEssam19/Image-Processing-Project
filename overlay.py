import numpy as np
import cv2 as cv


def draw_lanes(img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    color_img = np.zeros_like(img)

    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))

    cv.fillPoly(color_img, np.int_(points), (0, 200, 255))
    inv_perspective = inv_perspective_warp(color_img)
    inv_perspective = cv.addWeighted(img, 1, inv_perspective, 0.7, 0)
    return inv_perspective
