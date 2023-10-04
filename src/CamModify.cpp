#include "CamModify.h"

#include <opencv2/calib3d.hpp>

using namespace cv;

CamModify::CamModify(const ReadStereoFS& config_storage)
    : config_storage(config_storage), stream_l(VideoCapture(0)), stream_r(VideoCapture(2))
{ 
    if (!stream_l.isOpened()) {
        throw std::runtime_error(
            "Left camera is not opened. _ Constructor CamModify"
        );
    }

    if (!stream_l.isOpened()) {
        throw std::runtime_error(
            "Right camera is not opened. _ Constructor CamModify"
        );
    }

    stream_l.set(CAP_PROP_FRAME_WIDTH, 640);
    stream_l.set(CAP_PROP_FRAME_HEIGHT, 480);
    stream_l.set(CAP_PROP_FPS, 30.0);
    stream_r.set(CAP_PROP_FRAME_WIDTH, 640);
    stream_r.set(CAP_PROP_FRAME_HEIGHT, 480);
    stream_r.set(CAP_PROP_FPS, 30.0);
}

CamModify::~CamModify() { 
    stream_l.release();
    stream_r.release();
    left_stereo_map.first.release();
    left_stereo_map.second.release();
    right_stereo_map.first.release();
    right_stereo_map.second.release();
    image_l.release();
    image_r.release();
 }

std::pair<pairMatMat, pairMatMat> CamModify::undistortInfoMat() {
    initUndistortRectifyMap(
        config_storage.cameraMat_left(),
        config_storage.distCoeff_left(),
        config_storage.rectifyMat_left(),
        config_storage.projectionMat_left(),
        Size(640, 480),
        CV_32FC1,
        left_stereo_map.first,
        left_stereo_map.second
    );

    initUndistortRectifyMap(    
        config_storage.cameraMat_right(),
        config_storage.distCoeff_right(),
        config_storage.rectifyMat_right(),
        config_storage.projectionMat_right(),
        Size(640, 480),
        CV_32FC1,
        right_stereo_map.first,
        right_stereo_map.second
    );

    return std::make_pair(left_stereo_map, right_stereo_map);
}

pairMatMat CamModify::takePicture() {
    stream_l.read(image_l);
    stream_r.read(image_r);

    return std::make_pair(image_l, image_r);
}

// TODO Continuing CamModify class based on Distance-Estimation-and-Depth-Map-creation-using-OpenCV source. 