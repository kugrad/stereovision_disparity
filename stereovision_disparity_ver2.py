#! /usr/bin/env python3

import numpy as np
import cv2 as cv
import threading
import time

TEST = 1

lock_l = threading.Lock()
lock_r = threading.Lock()

# image_l: cv.UMat = None
# image_r: cv.UMat = None
image_l = None
imaeg_r = None

def coords_mouse_disp(event, x, y, flags, param):
    # print("callbakc happend")
    if event == cv.EVENT_LBUTTONDOWN:
        print("{}, {}".format(x, y))
        disp = param
        print("disparity: {}".format(disp[y, x]))

        #print x,y,disp[y,x],filteredImg[y,x]
        # average=0
        # for u in range (-1,2):
        #     for v in range (-1,2):
        #         average += disp[y+u,x+v]
        # average=average/9
        # Distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
        # Distance= np.around(Distance*0.01,decimals=2)
        # print('Distance: '+ str(Distance)+' m')

def readLeftImage(left_stream: cv.VideoCapture):
    global image_l
    fps_15 = True

    while True:
        if fps_15:
            lock_l.acquire()
            _, image_l = left_stream.read()
            lock_l.release()
            fps_15 = False
        else:
            _, _ = left_stream.read()
            fps_15 = True
        
        time.sleep(0.033)

def readRightImage(right_stream: cv.VideoCapture):
    global image_r
    fps_15 = True

    while True:
        if fps_15:
            lock_r.acquire()
            ret, image_r = right_stream.read()
            lock_r.release()
            fps_15 = False
        else:
            _, _ = right_stream.read()
            fps_15 = True

        time.sleep(0.033)

    
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

    # print(
    #     '''
    #     camera matrix left:
    #     {}
    #     camera matrix right:
    #     {}
    #     '''
    #     .format(camera_matrix_left, camera_matrix_right
    # ))

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

    window_block_size = 1
    min_disparity = 0
    # num_disparity = 130 - min_disparity
    num_disparity = 16*5
    smoothing_factor = 4

    stereo = cv.StereoSGBM_create(
                minDisparity=min_disparity,
                numDisparities=num_disparity,
                blockSize=window_block_size,
                P1=(8 * (window_block_size**2) * smoothing_factor),
                P2=(32 * (window_block_size**2) * smoothing_factor),
                # disp12MaxDiff=5,
                preFilterCap=25,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=1,
                mode=cv.StereoSGBM_MODE_SGBM_3WAY
                )

    stereoR = cv.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

    # WLS FILTER Parameters
    lmbda = 8000.0
    sigma = 1.5
 
    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    l_thread = threading.Thread(target=readLeftImage, args=(stream_left,))
    r_thread = threading.Thread(target=readRightImage, args=(stream_right,))
    l_thread.daemon = True
    r_thread.daemon = True
    l_thread.start()
    time.sleep(0.16)
    r_thread.start()

    kernel = np.ones((3, 3), np.uint8)

    cv.namedWindow('Filtered Color Depth')

    while True:

        lock_l.acquire()
        image_left = image_l
        lock_l.release()
        lock_r.acquire()
        image_right = image_r
        lock_r.release()

        if np.shape(image_left) == () or np.shape(image_right) == ():
            continue

        # cv.imshow("left", image_left)
        # cv.imshow("right", image_right)

        l_undis_img = cv.remap(image_left, map_leftx, map_lefty, cv.INTER_LINEAR, cv.BORDER_CONSTANT) # Scalar()
        r_undis_img = cv.remap(image_right, map_rightx, map_righty, cv.INTER_LINEAR, cv.BORDER_CONSTANT) # Scalar()

        
        gray_l = cv.cvtColor(l_undis_img, cv.COLOR_RGB2GRAY)
        gray_r = cv.cvtColor(r_undis_img, cv.COLOR_RGB2GRAY)

        # cv.imshow("before normalize", gray_l)

        # gray_l = cv.normalize(src=l_undis_img, dst=l_undis_img, beta=0, alpha=255.0, norm_type=cv.NORM_MINMAX)
        # gray_r = cv.normalize(src=r_undis_img, dst=r_undis_img, beta=0, alpha=255.0, norm_type=cv.NORM_MINMAX)

        # alpha = -0.5
        # gray_l = np.clip((1 + alpha) * gray_l - 128 * alpha, 0, 255).astype(np.uint8)
        # gray_r = np.clip((1 + alpha) * gray_r - 128 * alpha, 0, 255).astype(np.uint8)

        # GaussianBlur
        gray_l = cv.GaussianBlur(gray_l, (3, 3), 0, 0, borderType=cv.INTER_LINEAR)
        gray_r = cv.GaussianBlur(gray_r, (3, 3), 0, 0, borderType=cv.INTER_LINEAR)

        canny_l = cv.Canny(gray_l, 50, 200, L2gradient=True)
        canny_r = cv.Canny(gray_r, 50, 200, L2gradient=True)

        # cv.imshow("canny left gray", gray_l)
        # cv.imshow("canny right gray", gray_r)

        disp_l = stereo.compute(canny_l, canny_r)
        disp_r = stereoR.compute(canny_r, canny_l)

        # disp_l = np.int16(disp_l)
        # disp_r = np.int16(disp_r)

        # print(
        # '''
        # disparity shape: {}
        # disparity l:

        # {}

        # dispairty shape: {}
        # disparity r:

        # {}
        # '''.format(disp_l.shape, disp_l, disp_r.shape, disp_r))

        # cv.imshow("disp_l", disp_l)

        # Using the WLS filter
        filtered_img = wls_filter.filter(disp_l, gray_l, None, disp_r)
        filtered_img = cv.normalize(src=filtered_img, dst=filtered_img, beta=0, alpha=255, norm_type=cv.NORM_MINMAX)
        filtered_img = np.uint8(filtered_img)

        # disp = ((disp_l.astype(np.float32) / 16) - min_disparity) / num_disparity # Calculation allowing us to have 0 for the most distant object able to detect

    #    # Resize the image for faster executions
        # dispR = cv.resize(disp, None, fx=0.7, fy=0.7, interpolation=cv.INTER_AREA)

    #     # Filtering the Results with a closing filter
        # closing= cv.morphologyEx(disp, cv.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise) 

        # # Colors map
        # dispc = (closing - closing.min()) * 255
        # dispC = dispc.astype(np.uint8)                                   # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
        # disp_Color= cv.applyColorMap(dispC, cv.COLORMAP_OCEAN)         # Change the Color of the Picture into an Ocean Color_Map
        filt_Color= cv.applyColorMap(filtered_img, cv.COLORMAP_OCEAN) 

    #     # Show the result for the Depth_image
    #     # cv.imshow("filtered_img", filtered_img)
    #     # cv.imshow('Disparity', disp)
    #     # cv2.imshow('Closing',closing)
    #     # cv.imshow('Color Depth',disp_Color)
        cv.imshow('Filtered Color Depth', filt_Color)
        cv.setMouseCallback('Filtered Color Depth', coords_mouse_disp, disp_l)

        # depth = baseline (meter) * focal length (pixel) / disparity-value (pixel)

        if cv.waitKey(1) == 27:
            break

