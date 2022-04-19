import cv2 as cv
import numpy as np

from moviepy.editor import VideoFileClip

from curve_fitting import get_curve
from overlay import draw_lanes
from sliding_window import sliding_window
from calibration_distorstion_correction import undistort
from perspective_transform import perspective_warp
from edge_detection import edge_detection


def img_pipeline(img):
    img = undistort(img)
    img_ = edge_detection(img)
    img_ = perspective_warp(img_)
    out_img, curves, lanes, ploty = sliding_window(img_, draw_windows=False)
    curverad = get_curve(img, curves[0], curves[1])
    lane_curve = np.mean([curverad[0], curverad[1]])
    img = draw_lanes(img, curves[0], curves[1])

    font = cv.FONT_HERSHEY_SIMPLEX
    fontColor = (0, 0, 0)
    fontSize = 0.5
    cv.putText(img, 'Lane Curvature: {:.0f} m'.format(lane_curve), (570, 620), font, fontSize, fontColor, 2)
    cv.putText(img, 'Vehicle offset: {:.4f} m'.format(curverad[2]), (570, 650), font, fontSize, fontColor, 2)
    return img


def vid_pipeline(input_file, output_file):
    myclip = VideoFileClip(input_file)
    clip = myclip.fl_image(img_pipeline)
    clip.write_videofile(output_file, audio=False)


vid_pipeline("challenge_video.mp4", "out.mp4")
