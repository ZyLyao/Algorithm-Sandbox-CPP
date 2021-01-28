#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <chrono>

using namespace std;

int main() {
    auto start = chrono::steady_clock::now();
    float time_factor = 1e6;
    float gap_tolerance = 2.0; //meter
    float min_thres = 1.0; //meter
    float max_thres = 5.0; //meter

    auto t_elapsed = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();
    cout << "Time: Start reading matrix at: " << float(t_elapsed) / time_factor << endl;
    std::ifstream ifs("../../Data_folder/output_mtx_20210114211548.txt", std::ifstream::in);

    float x_range = 60.0; // hard-coded for this dataset
    float y_range = 80.0; // hard-coded for this dataset
    float z_range = 45.0; // hard-coded for this dataset
    float cell_size = 0.1; // hard-coded for this dataset
    size_t x_num = x_range / cell_size;
    size_t y_num = y_range / cell_size;
    size_t z_num = z_range / cell_size;
    cout << "size: x " << x_num << " ,y " << y_num << " ,z " << z_num << endl;

    std::string str;
    int cnt_line{0};
    Eigen::MatrixXf grid(x_num * z_num, y_num);
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


    return 0;
}
