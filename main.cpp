#include <stdio.h>
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

int main(int argc, char **argv) {
//    if ( argc != 2 )
//    {
//        printf("usage: DisplayImage.out <Image_Path>\n");
//        return -1;
//    }
//    Mat image;
//    image = imread( argv[1], 1 );
//    if ( !image.data )
//    {
//        printf("No image data \n");
//        return -1;
//    }
//    namedWindow("Display Image", WINDOW_AUTOSIZE );
//    imshow("Display Image", image);
//    waitKey(0);

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
    std::vector<std::vector<string>> lines;
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
    cv::Mat_<float> a(x_num, y_num); // create cv mat object

    eigen2cv(test_mtx, a); // convert eigen matrix to opencv mat
//    a.convertTo(a,CV_16F);
    cout << "a type: " << a.depth() << " " << a.channels() << endl;
//    a.setTo(0.5);
    cout << "Mat: " << a.rows << " " << a.cols << endl;
    namedWindow("Display Image", WINDOW_NORMAL);
    imshow("Display Image", a);
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

