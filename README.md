# project: Lane-Detection

## Description:
    It's a software pipeline to identify the lane boundaries in a video from a front-facing camera on a car. It tracks the lane lines and the 
position of the car from the center of the lane. It tracks the radius of curvature of the road too. 
We assumed that the camera is mounted at the center of the car, such that the lane center is the midpoint at the bottom of the image between the two lines we've detected. 
The offset of the lane center from the center of the image (converted from pixels to meters) is the distance from the center of the lane.

## Usage:
