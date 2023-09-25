#! /opt/homebrew/bin/python3

import cv2 as cv
import numpy as np

if __name__ == '__main__':
    fs = cv.FileStorage("./config/calib_storage.yaml", cv.FILE_STORAGE_READ)

    camera_matrix_left = fs.getNode("camera_matrix_left").mat()
    camera_matrix_right = fs.getNode("camera_matrix_right").mat()
    dist_coefficient_left = fs.getNode("dist_coefficient_left ").mat()
    dist_coefficient_right = fs.getNode("dist-coefficient_right").mat()
    rotation = fs.getNode("rotation").mat()
    translation = fs.getNode("translation").mat()
    essential = fs.getNode("essential").mat()
    fundamental = fs.getNode("fundamental").mat()
    rectification_left = fs.getNode("rectification_left").mat()
    rectification_right = fs.getNode("rectification_right").mat()
    projection_left = fs.getNode("projection_left").mat()
    projection_right = fs.getNode("projection_right").mat()

    fs.release()

    map_leftx, map_lefty = cv.initUndistortRectifyMap(camera_matrix_left,
                                                      dist_coefficient_left,
                                                      rectification_left,
                                                      projection_left,
                                                      (660, 480),
                                                      5 # CV_32FC1
                                                      )

    map_rightx, map_righty = cv.initUndistortRectifyMap(camera_matrix_right,
                                                        dist_coefficient_right,
                                                        rectification_right,
                                                        projection_right,
                                                        (660, 480),
                                                        5 # CV_32FC1
                                                        )

    stream_left = cv.VideoCapture("./data/left.mp4")
    stream_right = cv.VideoCapture("./data/right.mp4")

    stream_left.set(cv.CAP_PROP_FRAME_WIDTH, 660),
    stream_left.set(cv.CAP_PROP_FRAME_HEIGHT, 480),
    stream_left.set(cv.CAP_PROP_FPS, 30.0)
    stream_right.set(cv.CAP_PROP_FRAME_WIDTH, 660),
    stream_right.set(cv.CAP_PROP_FRAME_HEIGHT, 480),
    stream_right.set(cv.CAP_PROP_FPS, 30.0)

    while True:
        ret, left_img = stream_left.read()
        ret, right_img = stream_right.read()

        # cv.imshow("left_image", left_img)

        left_img = cv.cvtColor(left_img, cv.COLOR_RGB2GRAY)
        right_img = cv.cvtColor(right_img, cv.COLOR_RGB2GRAY)

        kernel_size = 3

        smooth_left = cv.GaussianBlur(left_img, (kernel_size, kernel_size), 1.5)
        smooth_right = cv.GaussianBlur(right_img, (kernel_size, kernel_size), 1.5)

        window_size = 9    
        left_matcher = cv.StereoSGBM_create(
	        numDisparities=96,
	        blockSize=7,
	        P1=8*3*window_size**2,
	        P2=32*3*window_size**2,
	        disp12MaxDiff=1,
	        uniquenessRatio=16,
	        speckleRange=2,
	        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
        )

        right_matcher = cv.ximgproc.createRightMatcher(left_matcher)

        wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(80000)
        wls_filter.setSigmaColor(1.2)

        disparity_left = np.int16(left_matcher.compute(smooth_left, smooth_right))
        disparity_right = np.int16(right_matcher.compute(smooth_right, smooth_left) )

        wls_image = wls_filter.filter(disparity_left, smooth_left, None, disparity_right)
        wls_image = cv.normalize(src=wls_image, dst=wls_image, beta=0, alpha=255, norm_type=cv.NORM_MINMAX)
        wls_image = np.uint8(wls_image)

        cv.imshow("disparity map", wls_image)

        if cv.waitKey(1) == 27:
            break
