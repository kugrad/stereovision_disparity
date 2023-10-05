#include <iostream>
#include <chrono>
#include <thread>

#include <opencv2/highgui/highgui.hpp>

#include "ReadStereoFS.h"
#include "CamModify.h"

using namespace std::chrono;

int main(int argc, char* argv[]){

    ReadStereoFS config_storage(CONFIG_DIR_PATH "calib_storage.yaml");
    CamModify core(config_storage);

    core.undistortInfoMat();

    while (true) {

        core.takePicture();
        core.undistortImage();
        core.imageCvt2Gray();
        core.makeDisparityImages();

        core.showResultImage();

        if (cv::waitKey(1) == 27)
            break;
    }

}

