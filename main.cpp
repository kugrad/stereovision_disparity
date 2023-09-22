#include <iostream>
#include <chrono>
#include <thread>

#include <opencv2/core/core.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/ximgproc/disparity_filter.hpp>

using namespace cv;
using namespace std::chrono;

int main(int argc, char* argv[]){

    FileStorage storage(CONFIG_DIR_PATH "calib_storage.yaml", FileStorage::READ);

    Mat camera_matrix_left = Mat(3, 3, CV_64FC1);
    Mat camera_matrix_right = Mat(3, 3, CV_64FC1);
    Mat dist_coefficient_left;
    Mat dist_coefficient_right;
    Mat rotaion;
    Mat translation;
    Mat essential;
    Mat fundamental;
    Mat rectification_left;
    Mat rectification_right;
    Mat projection_left;
    Mat projection_right;

    storage["camera_matrix_left"] >> camera_matrix_left; 
    storage["camera_matrix_right"] >> camera_matrix_right;
    storage["dist_coefficient_left"] >> dist_coefficient_left;
    storage["dist_coefficient_right"] >> dist_coefficient_right;
    storage["rotation"] >> rotaion;
    storage["translation"] >> translation;
    storage["essential"] >> essential;
    storage["fundamental"] >> fundamental;
    storage["rectification_left"] >> rectification_left;
    storage["rectification_right"] >> rectification_right;
    storage["projection_left"] >> projection_left;
    storage["projection_right"] >> projection_right;

    storage.release();


    Mat map_leftx, map_lefty;
    Mat map_rightx, map_righty;
    initUndistortRectifyMap(
        camera_matrix_left,
        dist_coefficient_left,
        rectification_left,
        projection_left,
        Size(640, 480),
        CV_32FC1,
        map_leftx,
        map_lefty
    );
    initUndistortRectifyMap(    
        camera_matrix_right,
        dist_coefficient_right,
        rectification_right,
        projection_right,
        Size(640, 480),
        CV_32FC1,
        map_rightx,
        map_righty
    );

    auto stream_left = VideoCapture("/dev/video2", CAP_V4L2);
    auto stream_right = VideoCapture("/dev/video4", CAP_V4L2);

    Mat left_img, right_img;
    Mat left_out, right_out;
    while (true) {
        auto start_time = high_resolution_clock::now();

        stream_left.read(left_img); 
        stream_right.read(right_img);

        cvtColor(left_img, left_img, COLOR_BGR2RGB);
        cvtColor(right_img, right_img, COLOR_BGR2RGB);

        remap(left_img, left_out, map_leftx, map_lefty, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        remap(right_img, right_out, map_rightx, map_righty, INTER_LINEAR, BORDER_CONSTANT, Scalar());

        imshow("left", left_out);
        imshow("right", right_out);


        // disparity map
        // char c = static_cast<char>(waitKey(1) & 0xff);
        // if (c == 0x43 || c == 0x63) {

            Mat gray_left, gray_right;

            cvtColor(left_img, gray_left, COLOR_RGB2GRAY);
            cvtColor(right_img, gray_right, COLOR_RGB2GRAY);

            int kernel_size = 7;

            // int ndisparities = 32;
            // int blocksize = 15;

            // // create stereo image
            // Mat img_disparity_16s, img_disparity_8u;

            // cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(ndisparities, blocksize);
            // stereo->compute(gray_left, gray_right, img_disparity_16s);

            // // normalize img_disparity_16s
            // double minVal; double maxVal;
            // minMaxLoc( img_disparity_16s, &minVal, &maxVal );
            // // cout << "min : " << minVal << ", max : " << maxVal << endl;
            // img_disparity_16s.convertTo(img_disparity_8u, CV_8UC1, 255/(maxVal - minVal));

            // // imshow("left", img1);
            // // imshow("right", img2);    
            // imshow("disparity", img_disparity_8u);

            Mat smooth_left, smooth_right;
            GaussianBlur(left_img, smooth_left, Size(kernel_size, kernel_size), 1, 5);
            GaussianBlur(right_img, smooth_right, Size(kernel_size, kernel_size), 1, 5);

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
                    cv::StereoSGBM::MODE_SGBM_3WAY /* mode */
                );

            Ptr<StereoMatcher> right_matcher = ximgproc::createRightMatcher(left_matcher);

            Ptr<ximgproc::DisparityWLSFilter> wls_filter = 
                ximgproc::createDisparityWLSFilter(left_matcher);

            wls_filter->setLambda(80000);
            wls_filter->setSigmaColor(1.2);

            Mat disparity_left;
            left_matcher->compute(left_img, right_img, disparity_left);
            disparity_left.convertTo(disparity_left, CV_16S);

            Mat disparity_right;
            right_matcher->compute(left_img, right_img, disparity_right);
            disparity_right.convertTo(disparity_right, CV_16S);

            Mat wls_image;
            wls_filter->filter(disparity_left, left_img, wls_image, disparity_right, cv::Rect(), right_img);
            cv::normalize(wls_image, wls_image, 255.0f, 0.0f, NORM_MINMAX);
            wls_image.convertTo(wls_image, CV_8U);

            imshow("dispairty map", wls_image);

            // // Mat imgDisparity16S = Mat(left_img.rows, left_img.cols, CV_16S);
            // // Mat imgDisparity8U = Mat(right_img.rows, right_img.cols, CV_8UC1);

            // // int ndisparities = 16*1;
            // // int SADWindowSize = 3;
        
            // // Ptr<StereoSGBM> sgbm = StereoSGBM::create(
            // //     0,
            // //     ndisparities,
            // //     SADWindowSize
            // // );
        
            // // sgbm->setP1(24*SADWindowSize*SADWindowSize);
            // // sgbm->setP2(96*SADWindowSize*SADWindowSize);
            // // sgbm->setPreFilterCap(63);
            // // sgbm->setMode(StereoSGBM::MODE_SGBM);
        
            // // sgbm->compute(gray_right, gray_left, imgDisparity16S);
            // // //imwrite( "test.jpg", imgDisparity16S );
        
            // // double minVal, maxVal;
        
            // // minMaxLoc( imgDisparity16S, &minVal, &maxVal);
        
            // // imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255/(maxVal - minVal));
            // // //imgDisparity16S.convertTo(imgDisparity8U, CV_32F, 1.0/16.0, 0.0);
            // // //imgDisparity16S.convertTo(imgDisparity8U, CV_8U);
        
            // // //namedWindow("windowDisparity", WINDOW_NORMAL);
            // // imshow("windowDisparity", imgDisparity8U);
            // // imshow("16S", imgDisparity16S);
        // }
        auto end_time = high_resolution_clock::now();
        duration<double> elapse =  end_time - start_time;

        double sleep_time = (1 / 30.f) - elapse.count();
        if (sleep_time > 0) {
            std::this_thread::sleep_for(microseconds(static_cast<int>(sleep_time * 1e6)));
        }


        if (waitKey(1) == 27)
            break;
    }

}

