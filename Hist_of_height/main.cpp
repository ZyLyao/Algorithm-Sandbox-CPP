#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <numeric>

using namespace std;
using namespace Eigen;
using namespace cv;

void drawHist(const vector<int>& data, Mat3b& dst, int binSize = 1, int height = 0)
{
    int max_value = *max_element(data.begin(), data.end());
    int rows = 0;
    int cols = 0;
    if (height == 0) {
        rows = max_value + 10;
    } else {
        rows = max(max_value + 10, height);
    }

    cols = data.size() * binSize;

    dst = Mat3b(rows, cols, Vec3b(0,0,0));


    for (int i = 0; i < data.size(); i++)
    {
        int h = rows - data[i];
        rectangle(dst, Point(i*binSize, h), Point((i + 1)*binSize-1, rows), (i%2) ? Scalar(0, 100, 255) : Scalar(0, 0, 255), FILLED);
    }
}

int main() {
    /** Import data*/
    auto start = chrono::steady_clock::now();
    float time_factor = 1e6;
    float gap_tolerance = 2.0; //meter
    float min_thres = 1.0; //meter
    float max_thres = 5.0; //meter

    auto t_elapsed = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();
    cout << "Time: Start reading matrix at: " << float(t_elapsed) / time_factor << endl;
    std::ifstream ifs("../../Data_folder/output_mtx_20210114211548.txt", std::ifstream::in);
//    std::ifstream ifs("../../Data_folder/output_mtx_dig.txt", std::ifstream::in);

    float x_range = 60.0; // hard-coded for this dataset
    float y_range = 80.0; // hard-coded for this dataset
    float z_range = 45.0; // hard-coded for this dataset
    float z_start = 15.0; // height where the grid starts
    float z_lim = 35.0f - z_start; // lowest possible height limitation of deck, for removing some pile points
    float cell_size = 0.1; // hard-coded for this dataset
    size_t x_num = x_range / cell_size;
    size_t y_num = y_range / cell_size;
    size_t z_num = z_range / cell_size;
    cout << "size: x " << x_num << " ,y " << y_num << " ,z " << z_num << endl;

    std::string str;
    int cnt_line{0};
    Eigen::MatrixXf grid(x_num * z_num, y_num);
    cout << "grid size: rows " << grid.rows()<<" , cols " << grid.cols() << endl;
    grid.setZero();
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
    t_elapsed = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();
    cout << "Time: Done reading matrix at: " << float(t_elapsed) / time_factor << endl;

    /** Convert the grid into a boolean grid*/
    long n_rows{grid.rows()}, n_cols{grid.cols()};
    Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic> grid_int;
    grid_int.resize(n_rows, n_cols);
    for (size_t i{0};i<grid.cols();i++){
        for (size_t j{0};j<grid.rows();j++){
            grid_int(j, i) = abs(grid(j, i)) > std::numeric_limits<float>::epsilon(); // if non-zero, set to true
        }
    }



    /** Calculate histogram of the height of points*/
    vector<int> hist(z_num, 0); // bins
    for (size_t i{0}; i < z_num; i++) {
        hist[i] = grid_int.block(i * x_num, 0, x_num, y_num).sum();
    }
    cout << "hist size: " << hist.size() << endl;
    Mat3b image;
    drawHist(hist, image);
    cout << "Mat size before resize: " << image.rows << " " << image.cols << endl;
    // Set height limitation for discarding some of the pile points
    int z_limit_pixel = static_cast<int>(z_lim / cell_size);
    line(image, Point(z_limit_pixel, 0), Point(z_limit_pixel, image.rows), Scalar(0, 255, 255), 1);
    cout << "z_limit_pixel: " << z_limit_pixel << endl;

    auto itr_hist_max = max_element(hist.begin(), hist.begin() + z_limit_pixel);
    int hist_max_idx = distance(hist.begin(), itr_hist_max);
    float hist_total = std::accumulate(hist.begin(), hist.begin() + z_limit_pixel, 0);
    cout << "hist max index: " << hist_max_idx << " Value: " << *itr_hist_max << " Total: " << hist_total << " Ratio: "
         << static_cast<float>(*itr_hist_max) / hist_total << endl;
    float h_hist_thres = 0.5f * *itr_hist_max;
    vector<pair<size_t, size_t>> h_hist_peaks;
    for (int i{0};i<hist.size();++i) {
        if(hist[i]>h_hist_thres && i < z_limit_pixel){
            h_hist_peaks.emplace(h_hist_peaks.end(), pair<size_t, size_t>(i, hist[i]));
            cout << "new peak: " << i << " " << hist[i] << "\n";
        }
    }
    cout << " Deck peak: " << h_hist_peaks.back().first<<" "<<h_hist_peaks.back().second << endl;

    line(image, Point(hist_max_idx, 0), Point(hist_max_idx, image.rows), Scalar(0, 255, 0), 1,LineTypes::LINE_4);

    /** Show image*/
    cv::resize(image, image, Size(960, 540));
    imshow("Histogram: Green Maximum; Yellow Z-limitation", image);












    waitKey();
    return 0;
}
