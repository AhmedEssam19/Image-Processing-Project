import pickle
import collections
import sys
from time import time
import cv2 as cv
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.ndimage import label
from moviepy.editor import VideoFileClip

from draw_labeled_bboxes import add_heat, apply_threshold, draw_labeled_bboxes
from find_car import find_car

dist_pickle = pickle.load(open("classifier_info.p", "rb"))
model = dist_pickle["svc"]
X_scaler_l = dist_pickle["scaler"]
orient_l = dist_pickle["orient"]
pix_per_cell_l = dist_pickle["pix_per_cell"]
cell_per_block_l = dist_pickle["cell_per_block"]
spatial_size_l = dist_pickle["spatial_size"]
hist_bins_l = dist_pickle["hist_bins"]


def img_pipeline_hog(img):
    heatmaps = collections.deque(maxlen=29)

    heat = np.zeros_like(img[:, :, 0]).astype(np.float)

    ystarts = [350]
    ystops = [570]

    # Look for cars at different scales
    scales = [1.5]

    for scale, ystart, ystop in zip(scales, ystarts, ystops):
        box_list, out_img, out_img_windows = find_car(img, ystart, ystop, scale, model, X_scaler_l, orient_l,
                                                      pix_per_cell_l, cell_per_block_l, spatial_size_l,
                                                      hist_bins_l)
        heat = add_heat(heat, box_list)
    # Append heatmap and compute the sum of the last n ones
    heatmaps.append(heat)
    sum_heatmap = np.array(heatmaps).sum(axis=0)
    # Apply the threshold to remove false positives
    heat = apply_threshold(sum_heatmap, min(len(heatmaps) * 1, 28))

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    return draw_img


detector = cv.dnn.readNetFromDarknet(darknetModel="yolov3.weights", cfgFile="yolov3.cfg")


def img_pipeline_yolo(image):
    h, w = image.shape[:2]
    layer_names = detector.getLayerNames()
    out_layer_names = [layer_names[i - 1] for i in detector.getUnconnectedOutLayers()]

    blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), crop=False, swapRB=False)
    detector.setInput(blob)

    layers_output = detector.forward(out_layer_names)
    boxes, confidences, classIDs = [], [], []
    for output in layers_output:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.5 and classID == 2:
                box = detection[:4] * np.array([w, h, w, h])
                bx, by, bw, bh = box.astype(int)
                x, y, = int(bx - bw / 2), int(by - bh / 2)
                boxes.append([x, y, bw, bh])
                confidences.append(confidence)
                classIDs.append(classID)

    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
    for idx in indices:
        x, y, bw, bh = boxes[idx]
        cv.rectangle(image, (x, y), (x + bw, y + bh), color=(0, 255, 255), thickness=2)

    return image


def vid_pipeline(input_file, output_file):
    myclip = VideoFileClip(input_file)
    myclip = myclip.set_fps(20)
    clip = myclip.fl_image(img_pipeline_yolo)
    clip.write_videofile(output_file, audio=False)


if __name__ == "__main__":
    # image = mpimg.imread('test_images/test1.jpg')
    # t = time()
    # img = img_pipeline(image)
    # print("total", time()-t)
    # plt.imshow(img)
    # plt.show()
    vid_pipeline(*sys.argv[1:])
