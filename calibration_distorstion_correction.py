import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pickle




## Calibration and Destorsion Correction of images
# helpful website https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# helpful video https://www.youtube.com/watch?v=3h7wgR5fYik&t=1352s
def undistort_img():
    # Prepare object points 0,0,0 ... 8,5,0 getting 3d points of the object
    # and the corressponding 2d point of the image that used in callibration
    #here we prepare the object points 
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # these two arrays to store object points and image points from all images
    objpoints = []
    imgpoints = []

    # we get all images to be calebrated from the path of directory
    images = glob.glob('camera_cal/*.jpg')

    #looping on the images to get image by image
    for indx, fname in enumerate(images):
        #get every image then store it in img variable
        img = cv2.imread(fname)
        #convert the image to gray scale as it based on one channel and easy in processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #use this function in opencv to detect the corners of the chessboard in the image and if
        #there is a chessboard in the img we make flag with truue and save the corners of chessboard in corners
        flag, corners = cv2.findChessboardCorners(gray, (9,6), None)

        if flag == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    #all of the previous code we try to get the chessboard borders and (objectpoints and image points)
    # get the frame size of the image
    frame_size = (img.shape[1], img.shape[0])
    #then we use calibratecamera from opencv to help us in calibration 
    # we pass to this func the opject points and image points we got and the frame size of the image
    #after using this function it will return a flag that tell us if there is callibration occur or not 
    #camera_matrix the result helps to transform 3d points to 2d points
    #dist is the distorsion attribute that helps in undistorsion function later
    flag, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_size, None,None)

    # as we want to undistort the image and remove destorsion from it as there is two methods 
    # using remapping 
    #or using undistort function in opencv which we pass to it distorsion factor and camera matrix that we got 
    # from calibration and passing also the original image
    dst = cv2.undistort(img, camera_matrix, dist, None, camera_matrix)
    # We create an object to save the result of calibration function as follow
    dist_pickle = {}
    dist_pickle['camera_matrix'] = camera_matrix
    dist_pickle['dist'] = dist
    pickle.dump(dist_pickle, open('camera_cal/cal_pickle.p', 'wb'))
    return dist_pickle

# we create this function based on all the previous to call it easly for making undistortion and callibration
def undistort(img, cal_dir='camera_cal/cal_pickle.p'):
    #cv2.imwrite('camera_cal/test_cal.jpg', dst)
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    camera_matrix = file['camera_matrix']
    dist = file['dist']
    dst = cv2.undistort(img, camera_matrix, dist, None, camera_matrix)
    
    return dst