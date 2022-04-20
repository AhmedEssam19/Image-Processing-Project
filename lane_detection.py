import numpy as np
import cv2
import cv2 as cv
import pickle
from moviepy.editor import VideoFileClip
import glob
import sys


def undistort_img():
    # Prepare object points 0,0,0 ... 8,5,0 getting 3d points of the object
    # and the corressponding 2d point of the image that used in callibration
    # here we prepare the object points
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # these two arrays to store object points and image points from all images
    objpoints = []
    imgpoints = []

    # we get all images to be calebrated from the path of directory
    images = glob.glob('camera_cal/*.jpg')

    # looping on the images to get image by image
    for indx, fname in enumerate(images):
        # get every image then store it in img variable
        img = cv2.imread(fname)
        # convert the image to gray scale as it based on one channel and easy in processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # use this function in opencv to detect the corners of the chessboard in the image and if
        # there is a chessboard in the img we make flag with truue and save the corners of chessboard in corners
        flag, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if flag:
            objpoints.append(objp)
            imgpoints.append(corners)
    # all of the previous code we try to get the chessboard borders and (objectpoints and image points)
    # get the frame size of the image
    frame_size = (img.shape[1], img.shape[0])
    # then we use calibratecamera from opencv to help us in calibration
    # we pass to this func the opject points and image points we got and the frame size of the image
    # after using this function it will return a flag that tell us if there is callibration occur or not
    # camera_matrix the result helps to transform 3d points to 2d points
    # dist is the distorsion attribute that helps in undistorsion function later
    flag, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_size, None, None)

    # as we want to undistort the image and remove destorsion from it as there is two methods
    # using remapping
    # or using undistort function in opencv which we pass to it distorsion factor and camera matrix that we got
    # from calibration and passing also the original image
    # We create an object to save the result of calibration function as follow
    dist_pickle = {'camera_matrix': camera_matrix, 'dist': dist}
    pickle.dump(dist_pickle, open('camera_cal/cal_pickle.p', 'wb'))
    return dist_pickle


# we create this function based on all the previous to call it easly for making undistortion and callibration
def undistort(img, cal_dir='camera_cal/cal_pickle.p'):
    # cv2.imwrite('camera_cal/test_cal.jpg', dst)
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    camera_matrix = file['camera_matrix']
    dist = file['dist']
    dst = cv2.undistort(img, camera_matrix, dist, None, camera_matrix)

    return dst


def color_filter(image):
    #convert to HLS to mask based on HLS
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lower = np.array([0,190,0])
    upper = np.array([255,255,255])
    yellower = np.array([10,0,90])
    yelupper = np.array([50,255,255])
    yellowmask = cv2.inRange(hls, yellower, yelupper)
    whitemask = cv2.inRange(hls, lower, upper)
    mask = cv2.bitwise_or(yellowmask, whitemask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked


def get_curve(img, leftx, rightx):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    y_eval = np.max(ploty)
    ym_per_pix = 30.5 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 720  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    car_pos = img.shape[1] / 2
    l_fit_x_int = left_fit_cr[0] * img.shape[0] ** 2 + left_fit_cr[1] * img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0] * img.shape[0] ** 2 + right_fit_cr[1] * img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
    center = (car_pos - lane_center_position) * xm_per_pix / 10
    # Now our radius of curvature is in meters
    return left_curverad, right_curverad, center


def edge_detection(img, filter_color, s_thresh=(100, 255), sx_thresh=(15, 255)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    # we use HLS color system to benifit us insted of rgb system as we geet in our testcases
    # some destortion in photos like sunlight and shadows ... and hls helps us to avoid these distortion
    # we will work on L_channel of the system
    if filter_color:
        img = color_filter(img)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # we use sobel x detection to detect the edges in the photo using l_channel of hls of our image

    sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)  # Take the derivative in x
    # after using sobel x we use the absolute of sobel x to get rid of gray color and clarify the
    # the white edges of the image
    absolute_sobelx = np.absolute(sobel_x)  # Absolute x derivative to accentuate lines away from horizontal
    # if we print (abs_sobelx) we find that the maximum number less than 255 then
    # we take scale to make the maximum value is 255
    scaled_sobel = np.uint8(255 * absolute_sobelx / np.max(absolute_sobelx))

    # Threshold x gradient
    # then we use binary image for more clarification of white edges in the photo

    # in this we take the photo which is scale_sobel then take numpy zeros of it
    sx_binary = np.zeros_like(scaled_sobel)
    # here we say if scaled_sobel bigger than minimum sobel threshold which is 15
    # and scaled sobel less than maximum sobel threshold which is 255 we but sx_binary by 1
    sx_binary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    combined_binary = np.zeros_like(sx_binary)
    combined_binary[(s_binary == 1) | (sx_binary == 1)] = 1
    return combined_binary


def get_hist(img):
    hist = np.sum(img[img.shape[0] // 2:, :], axis=0)
    return hist


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


def sliding_window(warped_img, nwindows=9):
    left_a, left_b, left_c = [], [], []
    right_a, right_b, right_c = [], [], []
    left_fit_ = np.empty(3)
    right_fit_ = np.empty(3)
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((warped_img, warped_img, warped_img)) * 255

    # Find the peak of the left and right halves of the histogram
    histogram = get_hist(warped_img)

    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(warped_img.shape[0] / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_img.shape[0] - (window + 1) * window_height
        win_y_high = warped_img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])

    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])

    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])

    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])

    ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty


def img_pipeline(original_image):
    img = undistort(original_image)

    try:
        img = edge_detection(img, filter_color=True)
        img = perspective_warp(img)
        out_img, curves, lanes, ploty = sliding_window(img)
    except:
        img = edge_detection(img, filter_color=False)
        img = perspective_warp(img)
        out_img, curves, lanes, ploty = sliding_window(img)

    curverad = get_curve(img, curves[0], curves[1])
    lane_curve = np.mean([curverad[0], curverad[1]])
    img = draw_lanes(img, curves[0], curves[1])
    # img[0:60, -60:0] = cv.resize(debug_images[0], (60, 60))
    font = cv.FONT_HERSHEY_SIMPLEX
    fontColor = (0, 0, 0)
    fontSize = 0.5
    cv.putText(img, 'Lane Curvature: {:.0f} m'.format(lane_curve), (570, 620), font, fontSize, fontColor, 2)
    cv.putText(img, 'Vehicle offset: {:.4f} m'.format(curverad[2]), (570, 650), font, fontSize, fontColor, 2)
    return img


def vid_pipeline(input_file, output_file, debug):
    myclip = VideoFileClip(input_file)
    clip = myclip.fl_image(img_pipeline)
    clip.write_videofile(output_file, audio=False)


if __name__ == "__main__":
    vid_pipeline(*sys.argv[1:])
