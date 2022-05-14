# project: Lane-Detection
	Detect lane lines using HSV filtering and sliding window search.

## Description:
	It's a software pipeline to identify the lane boundaries in a video from a front-facing camera on a car. It tracks the lane lines and the 
position of the car from the center of the lane. It tracks the radius of curvature of the road too. 
	We assumed that the camera is mounted at the center of the car, such that the lane center is the midpoint at the bottom of the image between the two lines we've detected. 
The offset of the lane center from the center of the image (converted from pixels to meters) is the distance from the center of the lane.

## Usage:
	It can be used in self driving cars. The car should follow the marked area to keep following the road and avoid hitting the sidewalk or other cars.

## How it works:
	The camera is mounted at the center of the car front. We work on each frame from the camera video.
	Each frame is first been calibrated to get rid of the distortion.
	Then it detects the edges in the frame so it can spot the lane lines.
	After detecting the edges, we cut the top part of the picture and use perspective warp so that we can get the bird view.
	Using the bird eye view, it detects the lane line and draw a line on in using the histogram and the sliding window show.
	This process is repeated for each frame.
