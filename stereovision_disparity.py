#! /usr/bin/env python3

import cv2 as cv
import numpy as np
import time

TEST = 1
STEREOSGBM = 1

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

    stream_left, stream_right = None, None
    if TEST:
        stream_left = cv.VideoCapture("./data/left.mp4")
        stream_right = cv.VideoCapture("./data/right.mp4")
    else:
        stream_left = cv.VideoCapture(2)
        stream_right = cv.VideoCapture(4)

    stream_left.set(cv.CAP_PROP_FRAME_WIDTH, 660),
    stream_left.set(cv.CAP_PROP_FRAME_HEIGHT, 480),
    stream_left.set(cv.CAP_PROP_FPS, 30.0)
    stream_right.set(cv.CAP_PROP_FRAME_WIDTH, 660),
    stream_right.set(cv.CAP_PROP_FRAME_HEIGHT, 480),
    stream_right.set(cv.CAP_PROP_FPS, 30.0)

    # Create StereoSGBM and prepare all parameters
    window_block_size = 9
    min_disparity = 2
    num_disparity = 130 - min_disparity

    stereo = None
    if STEREOSGBM:
        stereo = cv.StereoSGBM_create(
                    minDisparity=min_disparity,
                    numDisparities=num_disparity,
                    blockSize=window_block_size,
                    P1=(8 * 3 * window_block_size**2),
                    P2=(32 * 3 * window_block_size**2),
                    disp12MaxDiff=5,
                    preFilterCap=0,
                    uniquenessRatio=10,
                    speckleWindowSize=100,
                    speckleRange=32,
                    mode=cv.StereoSGBM_MODE_SGBM_3WAY
                    )
    else:
        stereo = cv.StereoBM_create(numDisparities=16, blockSize=21)

    # Used for the filtered image
    stereoR = cv.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

    # WLS FILTER Parameters
    lmbda = 80000
    sigma = 1.8
    visual_multiplier = 1.0
 
    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    kernel= np.ones((3,3),np.uint8)

    while True:
        ret, right_img = stream_right.read()
        ret, left_img = stream_left.read()

        # cv.imshow("left_image", left_img)

        # left_img = cv.cvtColor(left_img, cv.COLOR_BGR2RGB)
        # right_img = cv.cvtColor(right_img, cv.COLOR_BGR2RGB)

        # left_undistored_image = cv.remap(left_img, map_leftx, map_lefty, interpolation=cv.INTER_LANCZOS4, borderMode=cv.BORDER_CONSTANT) # Scalar()
        # right_undistored_image = cv.remap(right_img, map_rightx, map_righty, interpolation=cv.INTER_LANCZOS4, borderMode=cv.BORDER_CONSTANT) # Scalar()
        left_undistored_image = cv.remap(left_img, map_leftx, map_lefty, cv.INTER_LINEAR, cv.BORDER_CONSTANT) # Scalar()
        right_undistored_image = cv.remap(right_img, map_rightx, map_righty, cv.INTER_LINEAR, cv.BORDER_CONSTANT) # Scalar()

        gray_l = cv.cvtColor(left_undistored_image, cv.COLOR_RGB2GRAY)
        gray_r = cv.cvtColor(right_undistored_image, cv.COLOR_RGB2GRAY)

        cv.imshow("left gray", gray_l)
        cv.imshow("right gray", gray_r)

        # # GaussianBlur
        # gray_l = cv.GaussianBlur(gray_l, (3, 3), 0)
        # gray_r = cv.GaussianBlur(gray_r, (3, 3), 0)

        disp = stereo.compute(gray_l, gray_r)
        disp_l = disp
        disp_r = stereoR.compute(gray_r, gray_l)
        disp_l = np.int16(disp_l)
        disp_r = np.int16(disp_r)

        # Using the WLS filter
        filtered_img = wls_filter.filter(disp_l, gray_l, None, disp_r)
        filtered_img = cv.normalize(src=filtered_img, dst=filtered_img, beta=0, alpha=255, norm_type=cv.NORM_MINMAX)
        filtered_img = np.uint8(filtered_img)

        disp = ((disp.astype(np.float32) / 16) - min_disparity) / num_disparity # Calculation allowing us to have 0 for the most distant object able to detect

    #    # Resize the image for faster executions
        # dispR = cv.resize(disp, None, fx=0.7, fy=0.7, interpolation=cv.INTER_AREA)

        # Filtering the Results with a closing filter
        closing= cv.morphologyEx(disp, cv.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise) 

        # # Colors map
        dispc = (closing - closing.min()) * 255
        dispC = dispc.astype(np.uint8)                                   # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
        disp_Color= cv.applyColorMap(dispC, cv.COLORMAP_OCEAN)         # Change the Color of the Picture into an Ocean Color_Map
        filt_Color= cv.applyColorMap(filtered_img, cv.COLORMAP_OCEAN) 

        # Show the result for the Depth_image
        cv.imshow("filtered_img", filtered_img)
        # cv.imshow('Disparity', disp)
        # cv2.imshow('Closing',closing)
        # cv.imshow('Color Depth',disp_Color)
        cv.imshow('Filtered Color Depth', filt_Color)

        # kernel_size = 3

        # left_img = cv.GaussianBlur(left_img, (kernel_size, kernel_size), 0.5)
        # right_img = cv.GaussianBlur(right_img, (kernel_size, kernel_size), 0.5)

        # window_size = 1
        # # left_matcher = cv.StereoSGBM_create(
	    # #     # numDisparities=96,
	    # #     # blockSize=7,
	    # #     # P1=8*3*window_size**2,
	    # #     # P2=32*3*window_size**2,
	    # #     # disp12MaxDiff=1,
	    # #     # uniquenessRatio=16,
	    # #     # speckleRange=2,
	    # #     # mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
        # # )
        # # left_matcher = cv.StereoSBGM().create(0, 5*16, window_size)
        # left_matcher = cv.StereoSGBM().create(0, 5*16, window_size)
        # left_matcher.setP1(8*window_size*window_size)
        # left_matcher.setP2(96*window_size*window_size)
        # left_matcher.setPreFilterCap(63)
        # left_matcher.setMode(cv.STEREO_SGBM_MODE_SGBM_3WAY)

        # wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        # right_matcher = cv.ximgproc.createRightMatcher(left_matcher)

        # disparity_left = np.int16(left_matcher.compute(left_img, right_img))
        # disparity_right = np.int16(right_matcher.compute(right_img, left_img) )

        # wls_filter.setLambda(8000.0)
        # wls_filter.setSigmaColor(1.5)

        # # disparity_left = np.int16(left_matcher.compute(smooth_left, smooth_right))
        # # disparity_right = np.int16(right_matcher.compute(smooth_right, smooth_left) )

        # # wls_image = wls_filter.filter(disparity_left, smooth_left, None, disparity_right)
        # wls_image = wls_filter.filter(disparity_left, left_img, None, disparity_right)
        # wls_image = cv.normalize(src=wls_image, dst=wls_image, beta=0, alpha=255, norm_type=cv.NORM_MINMAX)
        # wls_image = np.uint8(wls_image)

        # cv.imshow("disparity map", wls_image)

        if cv.waitKey(1) == 27:
            break

stream_left.release()
stream_right.release()
cv.destroyAllWindows()