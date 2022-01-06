#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types.hpp>

std::vector<cv::String> filenames;
cv::Mat image;
std::vector<std::vector<cv::Point3f> > points3d;
std::vector<std::vector<cv::Point2f> > points2d;
std::vector<cv::Point2f> corner_points;
cv::Size image_size;
std::vector<cv::Point3f> point3d;
cv::Mat cameraMatrix,distCoeffs;
std::vector<cv::Mat> R,T;
std::vector<cv::Point2f>  imagePoints2;
std::vector<float> perViewErrors;
cv::Mat map1, map2;



void visualizeCorners()
{
    bool success;

    for (const auto& fn: filenames)
    {
        image = cv::imread(fn, cv::IMREAD_GRAYSCALE);
        success = cv::findChessboardCorners(image, cv::Size(6,5), corner_points, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
                                                                                 + cv::CALIB_CB_FAST_CHECK);
        if(success)
        {
            cornerSubPix(image, corner_points, cv::Size(11, 11), cv::Size(-1, -1),
                         cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT,30, 0.1));
            points3d.push_back(point3d);
            points2d.push_back(corner_points);
            image_size  = image.size();
        }
        drawChessboardCorners(image, cv::Size(6,5), cv::Mat(corner_points), success);
        cv::imshow("Corners",image);
        cv::waitKey(0);
    }
}

void ProjectionError()
{
    int i, totalPoints = 0;
    double totalErr = 0, err;
    double max=0, min=1;
    int max_index, min_index;

    perViewErrors.resize(points3d.size());
    for(i = 0; i < (int)points3d.size(); i++ )
    {
        projectPoints(cv::Mat(points3d[i]), R[i], T[i],cameraMatrix, distCoeffs, imagePoints2);
        err = norm(cv::Mat(points2d[i]), cv::Mat(imagePoints2), cv::NORM_L2);
        int n = (int)points3d[i].size();
        perViewErrors[i] = (float)std::sqrt(err*err/n);
        if(perViewErrors[i]>max)
        {
            max = perViewErrors[i];
            max_index = i;
        }
        if (perViewErrors[i]<min)
        {
            min=perViewErrors[i];
            min_index = i;
        }
        totalErr += err*err;
        totalPoints += n;
    }
    float x = std::sqrt(totalErr/totalPoints);
    std::cout<<"Projection Error"<<x<<std::endl;
    std::cout<<"max erro"<<max<<std::endl;
    std::cout<<"The name of image with max error"<<filenames[max_index]<<std::endl;
    std::cout<<"min error"<<min<<std::endl;
    std::cout<<"The name of image with min error"<<filenames[min_index]<<std::endl;
}

void Calliberation()
{
    cv::calibrateCamera(points3d,points2d,image_size, cameraMatrix, distCoeffs, R, T);
    std::cout << "Camera Matrix : " << cameraMatrix << std::endl;
    std::cout << "Distortion Coefficients : " << distCoeffs << std::endl;
//    for(int i = 0; i < (int)T.size(); i++ )
//    {
//        std::cout << "Rotation vector" <<i << ":" << R[i] << std::endl;
//        std::cout << "Translation vector" << i << ":" << T[i] << std::endl;
//    }
}

void undistorImage()
{
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs,
                                cv::Mat(), cameraMatrix, image_size,
                                CV_16SC2, map1, map2);
    int board_count = 0;  // resent max boards to read
    for (size_t i = 0; (i < filenames.size()) && (board_count < filenames.size()); ++i) {
        cv::Mat img, img0 = cv::imread(filenames[i]);
        ++board_count;
        if (!img0.data) {  // protect against no file
            std::cerr << filenames[i] << ", file #" << i << ", is not an img" << std::endl;
            continue;
        }

        cv::remap(img0, img, map1, map2, cv::INTER_LINEAR,
                  cv::BORDER_CONSTANT, cv::Scalar());
        cv::imshow("original undistorted", img);
        cv::imshow("original image", img0);
        cv::waitKey(0);
    }
}

int main()
{

    cv::utils::fs::glob(cv::String("../images"),cv::String("*.png"),filenames);
    for (int i=0; i<5;i++ ){
        for (int j=0; j<6;j++){
            point3d.push_back(cv::Vec3f(i*0.11, j*0.11,0.0));
        }
    }

    visualizeCorners();
    Calliberation();
    ProjectionError();
    undistorImage();

    cv::Mat image_result = cv::imread("../test_image.png",cv::IMREAD_COLOR);
    cv::Mat image_result0 = cv::imread("../test_image.png",cv::IMREAD_COLOR);
    cv::remap(image_result0, image_result, map1, map2, cv::INTER_LINEAR,
              cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    cv::imshow("original undistorted", image_result);
    cv::imshow("original image", image_result0);
    cv::waitKey(0);
    return 0;
}