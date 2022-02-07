#!/usr/bin/env python3

# USAGE
# python click_and_crop.py --image jurassic_park_kitchen.jpg --destination colors.txt

# import the necessary packages
import sys
import numpy as np
import cv2
import getopt


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global colorArray_bgr, colorArray_hsv, colorArray_yuv, image_bgr, image_hsv, image_yuv

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(x,y,image_bgr[y,x,0:3])
        colorArray_bgr = np.vstack([colorArray_bgr, image_bgr[y, x, 0:3]])
        colorArray_hsv = np.vstack([colorArray_hsv, image_hsv[y, x, 0:3]])
        colorArray_yuv = np.vstack([colorArray_yuv, image_yuv[y, x, 0:3]])

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        # print(x,y,image[y,x,0:3])
        colorArray_bgr = np.vstack([colorArray_bgr, image_bgr[y, x, 0:3]])
        colorArray_hsv = np.vstack([colorArray_hsv, image_hsv[y, x, 0:3]])
        colorArray_yuv = np.vstack([colorArray_yuv, image_yuv[y, x, 0:3]])

        # draw a rectangle around the region of interest
        # cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        # cv2.imshow("image", image)


def main(argv):
    global colorArray_bgr, colorArray_hsv, colorArray_yuv, image_bgr, image_hsv, image_yuv

    # initialize the list of reference points and boolean indicating
    # whether cropping is being performed or not
    colorArray_bgr = np.empty([1, 3])
    colorArray_hsv = np.empty([1, 3])
    colorArray_yuv = np.empty([1, 3])

    try:
        # USAGE
        # python click_and_crop.py --image jurassic_park_kitchen.jpg --destination colors.txt
        opts, args = getopt.getopt(argv, "i:d:", [])
    except getopt.GetoptError:
        print("wrong parameters")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--input"):
            input_image = arg
        elif opt in ("-d", "--destination"):
            destination_path = arg

    # load the image, clone it, and setup the mouse callback function
    dest_file = open(destination_path, "w")
    image_bgr = cv2.imread(input_image)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)

    clone = image_bgr.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image_bgr)
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the cropping region
        # if key == ord("r"):
        # 	image = clone.copy()

        # if the 'c' key is pressed, break from the loop
        if key == ord("c"):
            break

    # close all open windows
    colorArray_bgr = colorArray_bgr[1:, :]
    colorArray_hsv = colorArray_hsv[1:, :]
    colorArray_yuv = colorArray_yuv[1:, :]

    # print(colorArray.astype(int))
    print(
        "BGR Range ( %d:%d, %d:%d, %d:%d )"
        % (
            colorArray_bgr.min(axis=0)[0],
            colorArray_bgr.max(axis=0)[0],
            colorArray_bgr.min(axis=0)[1],
            colorArray_bgr.max(axis=0)[1],
            colorArray_bgr.min(axis=0)[2],
            colorArray_bgr.max(axis=0)[2],
        )
    )
    print(
        "HSV Range ( %d:%d, %d:%d, %d:%d )"
        % (
            colorArray_hsv.min(axis=0)[0],
            colorArray_hsv.max(axis=0)[0],
            colorArray_hsv.min(axis=0)[1],
            colorArray_hsv.max(axis=0)[1],
            colorArray_hsv.min(axis=0)[2],
            colorArray_hsv.max(axis=0)[2],
        )
    )
    print(
        "YUV Range ( %d:%d, %d:%d, %d:%d )"
        % (
            colorArray_yuv.min(axis=0)[0],
            colorArray_yuv.max(axis=0)[0],
            colorArray_yuv.min(axis=0)[1],
            colorArray_yuv.max(axis=0)[1],
            colorArray_yuv.min(axis=0)[2],
            colorArray_yuv.max(axis=0)[2],
        )
    )
    dest_file.write(
        "BGR Range ( %d:%d, %d:%d, %d:%d )\n"
        % (
            colorArray_bgr.min(axis=0)[0],
            colorArray_bgr.max(axis=0)[0],
            colorArray_bgr.min(axis=0)[1],
            colorArray_bgr.max(axis=0)[1],
            colorArray_bgr.min(axis=0)[2],
            colorArray_bgr.max(axis=0)[2],
        )
    )
    dest_file.write(
        "HSV Range ( %d:%d, %d:%d, %d:%d )\n"
        % (
            colorArray_hsv.min(axis=0)[0],
            colorArray_hsv.max(axis=0)[0],
            colorArray_hsv.min(axis=0)[1],
            colorArray_hsv.max(axis=0)[1],
            colorArray_hsv.min(axis=0)[2],
            colorArray_hsv.max(axis=0)[2],
        )
    )
    dest_file.write(
        "YUV Range ( %d:%d, %d:%d, %d:%d )"
        % (
            colorArray_yuv.min(axis=0)[0],
            colorArray_yuv.max(axis=0)[0],
            colorArray_yuv.min(axis=0)[1],
            colorArray_yuv.max(axis=0)[1],
            colorArray_yuv.min(axis=0)[2],
            colorArray_yuv.max(axis=0)[2],
        )
    )
    dest_file.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1:])
