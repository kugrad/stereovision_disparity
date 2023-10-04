#ifndef __CAM_MODIFY_H__
#define __CAM_MODIFY_H__

#include "ReadStereoFS.h"

#include <opencv2/videoio/videoio.hpp>

typedef std::pair<cv::Mat, cv::Mat> pairMatMat;

class CamModify {
public:
    CamModify(const ReadStereoFS& config_storage);
    ~CamModify();

    std::pair<pairMatMat, pairMatMat> undistortInfoMat();
    pairMatMat takePicture();

private:
    const ReadStereoFS config_storage;

    cv::VideoCapture stream_l, stream_r;

    pairMatMat left_stereo_map;
    pairMatMat right_stereo_map;

    cv::Mat image_l, image_r;

};

#endif