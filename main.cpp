#include <iostream>
#include <chrono>
#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>

#include "ReadStereoFS.h"
#include "CamModify.h"

using namespace cv;
using namespace std::chrono;

int main(int argc, char* argv[]){

    ReadStereoFS config_storage(CONFIG_DIR_PATH "calib_storage.yaml");
    CamModify core(config_storage);

    core.undistortInfoMat();

    while (true) {

        core.takePicture();

        // TODO Currently doin' here. Go to sleep..~~
        // ! checkpoint !

        cvtColor(left_img, left_img, COLOR_RGB2GRAY);
        cvtColor(right_img, right_img, COLOR_RGB2GRAY);

        // imshow("left_image", left_img);

        int kernel_size = 3;

        Mat smooth_left, smooth_right;
        GaussianBlur(left_img, smooth_left, Size(kernel_size, kernel_size), 1.5);
        GaussianBlur(right_img, smooth_right, Size(kernel_size, kernel_size), 1.5);

        int window_size = 9;
        Ptr<StereoSGBM> left_matcher =
            StereoSGBM::create(
                0,  /* min disparity */ 
                96, /* num disparity */
                7,  /* block size */
                8 * 3 * pow(window_size, 2), /* P1 */
                32 * 3 * pow(window_size, 2), /* p2 */
                1,  /* disp12MaxDiff */
                0,  /* preFilterCap */
                16, /* uniquenessRatio */
                0,  /* speckleWindowSize */
                2,  /* speckleRange */
                StereoSGBM::MODE_SGBM_3WAY /* mode */
            );

        auto right_matcher = ximgproc::createRightMatcher(left_matcher);

        auto wls_filter = ximgproc::createDisparityWLSFilter(left_matcher);

        wls_filter->setLambda(80000);
        wls_filter->setSigmaColor(1.2);

        Mat disparity_left;
        left_matcher->compute(smooth_left, smooth_right, disparity_left);
        disparity_left.convertTo(disparity_left, CV_16S);

        Mat disparity_right;
        right_matcher->compute(smooth_right, smooth_left, disparity_right);
        disparity_right.convertTo(disparity_right, CV_16S);

        Mat wls_image;
        // wls_filter->filter(disparity_left, smooth_left, wls_image, disparity_right, cv::Rect(), smooth_right);
        wls_filter->filter(disparity_left, smooth_left, wls_image, disparity_right);
        cv::normalize(wls_image, wls_image, 255.0f, 0.0f, NORM_MINMAX);
        // cv::normalize(wls_image, wls_image, 255.0, NORM_MINMAX);
        wls_image.convertTo(wls_image, CV_8U);

        imshow("dispairty map", wls_image);

        if (waitKey(1) == 27)
            break;
    }

}

