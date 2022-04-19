import numpy as np
import cv2

#--> using undistort function that we implement it in callibration file 

def edge_detection(img, s_thresh=(100, 255), sx_thresh=(15, 255)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel 
    # we use HLS color system to benifit us insted of rgb system as we geet in our testcases 
    # some destortion in photos like sunlight and shadows ... and hls helps us to avoid these distortion
    # we will work on L_channel of the system 

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # we use sobel x detection to detect the edges in the photo using l_channel of hls of our image

    sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1) # Take the derivative in x
    #after using sobel x we use the absolute of sobel x to get rid of gray color and clarify the 
    # the white edges of the image
    absolute_sobelx = np.absolute(sobel_x) # Absolute x derivative to accentuate lines away from horizontal
    #if we print (abs_sobelx) we find that the maximum number less than 255 then 
    # we take scale to make the maximum value is 255
    scaled_sobel = np.uint8(255*absolute_sobelx/np.max(absolute_sobelx))
    
    # Threshold x gradient
    #then we use binary image for more clarification of white edges in the photo

    # in this we take the photo which is scale_sobel then take numpy zeros of it
    sx_binary = np.zeros_like(scaled_sobel)
    # here we say if scaled_sobel bigger than minimum sobel threshold which is 15 
    # and scaled sobel less than maximum sobel threshold which is 255 we but sx_binary by 1
    sx_binary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    color_binary = np.dstack((np.zeros_like(sx_binary), sx_binary, s_binary)) * 255
    
    combined_binary = np.zeros_like(sx_binary)
    combined_binary[(s_binary == 1) | (sx_binary == 1)] = 1
    return combined_binary