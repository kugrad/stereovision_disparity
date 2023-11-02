#ifndef __CAM_MODIFY_H__
#define __CAM_MODIFY_H__

#include "ReadStereoFS.h"

#include <opencv2/videoio/videoio.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/calib3d.hpp>

#define TEST 0
#define STEREOSGBM 1

typedef std::pair<cv::Mat, cv::Mat> pairMatMat;

class CamModify {
public:
    CamModify(const ReadStereoFS& config_storage);
    ~CamModify();

    std::pair<pairMatMat, pairMatMat> undistortInfoMat();
    pairMatMat takePicture();
    pairMatMat undistortImage();
    pairMatMat imageCvt2Gray();
    pairMatMat makeDisparityImages();
    cv::Mat filterStereoImage();
    void showResultImage();
    // cv::Mat calculate3DCoordinate();



private:
    const ReadStereoFS config_storage;

    cv::VideoCapture stream_l, stream_r;

    pairMatMat left_stereo_map;
    pairMatMat right_stereo_map;

    cv::Mat image_l, image_r;
    cv::Mat gray_l, gray_r;
    cv::Mat disp_l, disp_r;
    cv::Mat filtered_img;
    cv::Mat filtered_img_colored;
    // cv::Mat point_cloud;

#if STEREOSGBM
    cv::Ptr<cv::StereoSGBM> stereo;
#else
    cv::Ptr<cv::StereoBM> stereo;
#endif
    cv::Ptr<cv::StereoMatcher> stereoR;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;


#if STEREOSGBM
    // Stereo SBGM parameter value
    constexpr static int smoothing_factor = 4;
    constexpr static int window_block_size = 3;
    constexpr static int min_disparity = 0;
    constexpr static int num_disparity = 16 * 5; 
    constexpr static int P1 = 8 * (window_block_size * window_block_size) * smoothing_factor;
    constexpr static int P2 = 32 * (window_block_size * window_block_size) * smoothing_factor;
    constexpr static int disp12MaxDiff = 5;
    constexpr static int preFilterCap = 25;
    constexpr static int uniquenessRatio = 10;
    constexpr static int speckleWindowSize = 100;
    constexpr static int speckleRange = 1;
    constexpr static int mode = cv::StereoSGBM::MODE_SGBM_3WAY;
#else
    const static int num_disparity = 16;
    const static int window_block_size = 21;
#endif

    // wls filter parameter value
    constexpr static double lmbda = 8000.0;
    constexpr static double sigma = 1.5;

};

#endif