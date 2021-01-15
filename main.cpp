#include <cstdio>
#include <opencv2/opencv.hpp>

// ifstream constructor.
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <opencv2/core/eigen.hpp>

using namespace cv;
using namespace std;

void get_z_vector(const Eigen::MatrixXf &grid, Eigen::VectorXf &vec, size_t x_idx, size_t y_idx, size_t step);

int main() {
//    std::ifstream ifs ("output.csv", std::ifstream::in);

//    std::ifstream ifs ("aa.txt", std::ifstream::in);
    std::ifstream ifs("output_mtx.txt", std::ifstream::in);
    float x_range = 60.0;
    float y_range = 80.0;
    float z_range = 45.0;
    float cell_size = 0.5;
    size_t x_num = x_range / cell_size;
    size_t y_num = y_range / cell_size;
    size_t z_num = z_range / cell_size;
    cout << "size: x " << x_num << " ,y " << y_num << " ,z " << z_num << endl;

    std::string str;
    Eigen::MatrixXf grid(10800, 160);
    grid.setZero();
    int cnt_line = 0;
    while (std::getline(ifs, str)) {
        std::vector<string> vec_this_line;
        boost::algorithm::split(vec_this_line, str, [](char c) { return c == ','; });
        for (size_t j = 0; j < vec_this_line.size(); j++) {
            grid(cnt_line, j) = stof(vec_this_line[j]);
//            grid(cnt_line, j) = (stof(vec_this_line[j]) > 0.1);
        }
        cnt_line++;
    }
    ifs.close();

    /** After this line is the same for SHU and this demo*/
    /** convert eigen matrix to Mat*/
    cout << "grid sum; " << grid.sum() << endl;
    Eigen::MatrixXf test_mtx(x_num, y_num);
    test_mtx.setZero();
    Eigen::VectorXf vec(z_num);
//    cout << "vec size: " << vec.size() << endl;
    for (size_t i{0}; i < x_num; i++) {
        for (size_t j{0}; j < y_num; j++) {
            get_z_vector(grid, vec, i, j, x_num);
            float max = -numeric_limits<float>::infinity();
            float min = numeric_limits<float>::infinity();
            for (size_t i_p{0}; i_p < vec.size(); i_p++) {
                if (vec[i_p] > max && vec[i_p] != 0) max = vec[i_p];
                if (vec[i_p] < min && vec[i_p] != 0) min = vec[i_p];
            }


            test_mtx(i, j) = max - min;

        }
    }


//    test_mtx.setRandom();
    cout << "test-Mtx: " << test_mtx << endl;
    cv::Mat_<float> pc_image(x_num, y_num); // create cv mat object

    eigen2cv(test_mtx, pc_image); // convert eigen matrix to opencv mat
//    pc_image.convertTo(pc_image,CV_8U);
    cout << "pc_image type: " << pc_image.depth() << " " << pc_image.channels() << endl;

    Mat pc_iamge_grey;
    pc_image.convertTo(pc_iamge_grey, CV_8U);

    cout << "Mat: " << pc_image.rows << " " << pc_image.cols << endl;
    namedWindow("Display Image", WINDOW_NORMAL);
    imshow("Display Image", pc_image);

    Mat image_edges;
    int lowThreshold = 0;
    const int ratio = 3;
    const int kernel_size = 3;

    Canny(pc_iamge_grey, image_edges, lowThreshold, lowThreshold * ratio, kernel_size);
    namedWindow("image_edges Image", WINDOW_NORMAL);
    imshow("image_edges Image", image_edges);

    vector<Vec3f> HT_v_lines;
//    vector<Vec4i> HT_v_lines;
    vector<Vec3f> HT_h_lines; // will hold the results of the detection
    cout << "num lines: " << HT_v_lines.size() << endl;

    float HT_theta_resolution = 0.1 * CV_PI / 180; // radian
    float HT_rho_resolution = 0.5f * cell_size; // pixel
    float min_edge_len = 10.0; // meter
    int HT_min_thres_x = int(min_edge_len / cell_size);
    int HT_min_thres_y = 3 * int(min_edge_len / cell_size);
    Mat cimage_edges;
    cvtColor(image_edges, cimage_edges, COLOR_GRAY2BGR);

    /** Find vertical lines*/
    HoughLines(image_edges, HT_v_lines, HT_rho_resolution, HT_theta_resolution, HT_min_thres_x, 0, 0, CV_PI / 180 * 0.1,
               CV_PI / 180 * 1); // runs the actual detection
    cout << "num lines: " << HT_v_lines.size() << endl;
    for (size_t i{0}; i < HT_v_lines.size(); i++) {
        cout << "v_line " << i << ": rho " << HT_v_lines[i][0] << ", theta " << HT_v_lines[i][1] / CV_PI * 180.0
             << ", cnt "
             << HT_v_lines[i][2] << endl;
    }

//    HoughLinesP(image_edges, HT_v_lines, HT_rho_resolution, HT_theta_resolution, HT_min_thres_x,HT_min_thres_x,3);
//    for( size_t i = 0; i < HT_v_lines.size(); i++ )
//    {
//        line( cimage_edges, Point(HT_v_lines[i][0], HT_v_lines[i][1]),
//              Point( HT_v_lines[i][2], HT_v_lines[i][3]), Scalar(255,0,0), 1, 8 );
//    }

    // Draw the lines
    for (auto &HT_v_line : HT_v_lines) {
        float rho = HT_v_line[0], theta = HT_v_line[1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(cimage_edges, pt1, pt2, Scalar(255, 0, 0), 1, LINE_AA);
    }

    /** Find vertical lines*/
    HoughLines(image_edges, HT_h_lines, HT_rho_resolution, HT_theta_resolution, HT_min_thres_y, 0, 0,
               CV_PI / 180 * 89,
               CV_PI / 180 * 91); // runs the actual detection
    for (size_t i{0}; i < HT_h_lines.size(); i++) {
        cout << "h_line " << i << ": rho " << HT_h_lines[i][0] << ", theta " << HT_h_lines[i][1] / CV_PI * 180.0
             << ", cnt "
             << HT_h_lines[i][2] << endl;
    }
    // Draw the lines
    for (auto &HT_h_line : HT_h_lines) {
        float rho = HT_h_line[0], theta = HT_h_line[1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(cimage_edges, pt1, pt2, Scalar(0, 0, 255), 1, LINE_AA);
    }

    namedWindow("Detected Lines  - Standard Hough Line Transform", WINDOW_NORMAL);
    imshow("Detected Lines  - Standard Hough Line Transform", cimage_edges);

    waitKey(0);
    return 0;

}

/**
 * @brief select voxels at same x,y indices
 * @param grid grid of the point cloud
 * @param vec a vector of voxels at same x,y indicess
 * @param x_idx target x
 * @param y_idx target y
 * @param step step size of x index from one z to the next z
 */
void get_z_vector(const Eigen::MatrixXf &grid, Eigen::VectorXf &vec, size_t x_idx, size_t y_idx, size_t step) {
//    cout << "get_z_vector: " << step << endl;
    for (size_t i{0}; i < vec.size(); i++) {
        vec[i] = grid(x_idx + i * step, y_idx);
    }
}

