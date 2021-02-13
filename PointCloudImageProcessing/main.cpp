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

void get_z_std_vector(const Eigen::MatrixXf &grid, std::vector<bool> &vec, size_t x_idx, size_t y_idx, size_t step) {
    size_t x_ = x_idx;
    for (size_t i{0}; i < vec.size()-1; i++)
    {
//        vec[i] = abs(grid(x_, y_idx)) > FLT_EPSILON;
//        cout << x_ - 1<<" "<< y_idx - 1 << endl;
        vec[i] = !grid.block(x_ - 1, y_idx - 1, 3, 3).isZero();
        x_ += step;
    }
}

pair<float, float>
approx_deck_Z_range(const Eigen::MatrixXf &grid, float margin, int x_num, int y_num, int z_num, float zmin,
                    float cell_size) {

    /** Convert the bool grid into a int grid*/
    long n_rows{grid.rows()}, n_cols{grid.cols()};
    Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic> grid_int;
    grid_int.resize(n_rows, n_cols);
    for (size_t i{0}; i < grid_int.cols(); i++) {
        for (size_t j{0}; j < grid_int.rows(); j++) {
            grid_int(j, i) = abs(grid(j, i)) > std::numeric_limits<float>::epsilon() ? 1 : 0; // if true, set to 1
        }
    }

    /** Calculate histogram of the height of points*/
    vector<int> hist(z_num, 0); // bins
    for (int i{0}; i < z_num; i++) {
        hist[i] = grid_int.block(i * x_num, 0, x_num, y_num).sum();
    }

    float z_start = zmin; // height where the grid_struct starts

    float z_lim = 35.0f - z_start; // lowest possible height limitation of deck
    size_t z_limit_pixel = static_cast<int>(z_lim / cell_size);

    auto itr_hist_max = max_element(hist.begin(), hist.begin() + z_limit_pixel);
    float h_hist_thres = 0.5f * (*itr_hist_max);
    vector<pair<size_t, size_t>> h_hist_peaks;
    for (size_t i{0}; i < hist.size(); ++i) {
        if (hist[i] > h_hist_thres && i < z_limit_pixel) {
            h_hist_peaks.emplace(h_hist_peaks.end(), pair<size_t, size_t>(i, hist[i]));
        }
    }
    size_t deck_idx = h_hist_peaks.back().first;

    /** Calculate the height range of the deck*/
    float h_deck = zmin + deck_idx * cell_size;
    return pair<float, float>(h_deck - margin, h_deck + margin);
}

#include <vector>
#include <utility>
#include <iostream>

std::pair<std::size_t, std::size_t>
get_bounded_interval(std::vector<bool> const &vals,
                     std::size_t const begin_idx)
// returns a bounding interval where the first and second indexes have
// equivalent values in vals and all indexes between first and second
// are not equal to the values at the first and second indexes. Note
// that the vals[begin_idx] value is used to determine the bounding
// value and all elements in the middle will not equal the bounding
// value. Some examples of bounded intervals are 1, 11, 101, 1001...
{
    std::size_t const SIZE_VALS{ vals.size() };
    std::size_t end_idx{ begin_idx + 1 };
    while(end_idx < SIZE_VALS)
    {
        if(vals[begin_idx] == vals[end_idx])
        {
            return { begin_idx, end_idx };
        }
        ++end_idx;
    }
    return { begin_idx, begin_idx };
}

std::size_t
len_gap(std::pair<std::size_t, std::size_t> const & gap)
// returns the length of a gap pair where the first and second values of the
// gap represent indexes. len_gap({0, 0}) = 0, len_gap({0, 1}) = 1, ...
{
    return std::max(gap.first, gap.second)
           - std::min(gap.first, gap.second);
}

std::vector<std::pair<std::size_t, std::size_t>>
max_gap(std::vector<bool> const & vals,
        bool const gap_val)
// returns indicies of the maximum gap of gap_vals discovered in vals. A gap
// is considered to be a continuous run of gap_vals bounded by not gap_vals.
{
    std::vector<std::pair<size_t, size_t>> max_gaps;
    std::size_t const SIZE_VALS{ vals.size() };
    std::size_t j{ 0U };
    std::size_t max_gap_len{ 0U };
    while(j < SIZE_VALS)
    {
        if(vals[j] == gap_val)
        {
            ++j;    // advance over gaps
            continue;
        }
        else
        {
            // linear search through vals for maximum gaps
            std::pair<std::size_t, std::size_t> gap{ get_bounded_interval(vals, j) };
            std::size_t const cur_gap_len{ len_gap(gap) };
            if(max_gaps.empty())
            {
                max_gap_len = cur_gap_len;
                max_gaps.push_back(gap);
            }
            else
            {
                if(cur_gap_len > max_gap_len)
                {
                    max_gap_len = cur_gap_len;
                    max_gaps.clear();
                    max_gaps.push_back(gap);
                }
                else if(cur_gap_len == max_gap_len)
                {
                    max_gaps.push_back(gap);
                }
            }
            if(gap.second == j)
            {
                ++j;    // degenerate gap
            }
            else
            {
                j = gap.second;
            }
        }
    }
    return max_gaps;
}

bool
get_max_gap(std::vector<bool> const & vals,
            std::size_t & max_gap_len,
            std::vector<std::pair<std::size_t, std::size_t>> & max_gap_idx_ranges,
            bool const gap_val)
// Determines the maximum gap in vals, where a gap is a continuous run of
// gap_val values beginning and ending with not gap_val values.
{
    max_gap_idx_ranges = max_gap(vals, gap_val);
    if(max_gap_idx_ranges.empty())
    {
        max_gap_len = 0U;
    }
    else
    {
        max_gap_len = len_gap(max_gap_idx_ranges.front());
    }
    return max_gap_len > 0U;
}

std::vector<std::pair<std::size_t, std::size_t>>
max_gap_with_holes(std::vector<bool> const & vals,
                   std::size_t const max_hole_len,
                   bool const hole_val)
// returns a vector of maximum gap with hole(s) where each hole does not
// exceed the max_hole_len, and a hole is a continuous run of hole_vals bounded
// by not hole_val's. For example a vals == 10011010 with max_hole_len 3 and
// hole_val == 0 will return { { 0, 6 } }; the trailing 0 is excluded.
{
    std::size_t const SIZE_VALS{ vals.size() };
    std::vector<std::pair<std::size_t, std::size_t>> max_gaps_with_holes;
    if(!SIZE_VALS)
    {
        return max_gaps_with_holes;
    }
    std::size_t k{ 0U };
    std::size_t max_gap_len{ 0U };
    while(k < SIZE_VALS)
    {
        if(vals[k] == hole_val)
        {
            ++k;    // advance over holes
            continue;
        }
        else
        {
            // greedily maximize the cur_gap by iteratively advancing
            // the size of cur_gap until falling off the end of the vals vector
            // or not finding a larger gap or discovering a gap greater than
            // the max_hole_len.
            std::pair<std::size_t, std::size_t> cur_gap{ k, k };
            std::pair<std::size_t, std::size_t> next_gap{ get_bounded_interval(vals, cur_gap.second) };
            std::size_t next_gap_len{ len_gap(next_gap) };
            while(next_gap.second < SIZE_VALS
                  && next_gap_len > 0
                  && next_gap_len <= max_hole_len)
            {
                cur_gap.second = next_gap.second;
                next_gap = get_bounded_interval(vals, cur_gap.second);
                next_gap_len = len_gap(next_gap);
            }
            if(max_gaps_with_holes.empty())
            {
                max_gap_len = len_gap(cur_gap);
                max_gaps_with_holes.push_back(cur_gap);
            }
            else
            {
                std::size_t const cur_gap_len{ len_gap(cur_gap) };
                if(cur_gap_len > max_gap_len)
                {
                    max_gap_len = cur_gap_len;
                    max_gaps_with_holes.clear();
                    max_gaps_with_holes.push_back(cur_gap);
                }
                else if(cur_gap_len == max_gap_len)
                {
                    max_gaps_with_holes.push_back(cur_gap);
                }
            }
            if(next_gap.second == k)
            {
                ++k;    // degenerate gap discovered, move on.
            }
            else
            {
                k = next_gap.second;
            }
        }
    }
    return max_gaps_with_holes;
}

bool
get_max_gap_with_holes(std::vector<bool> const & vals,
                       std::size_t & max_gap_with_holes_len,
                       std::vector<std::pair<std::size_t, std::size_t>> & max_gap_with_holes_idx_ranges,
                       std::size_t const max_hole_len,
                       bool const hole_val)
// returns true when there exists a maximum gap with holes in vals with size
// greater than zero. The discovered max gap with holes length is output
// through the max_gap_with_holes_len variable. A collection of max gaps with
// holes index ranges are output through the max_gap_with_holes_idx_ranges
// variable. Note that multiple max gaps with holes index ranges may be returned
// if there exits multiple ranges of equivalent maximum length. NOTE : there is
// a subtle distinction between a gap and a gap_with_holes. See max_gap(...) for
// the defition of a gap. A gap_with_holes is a bounded interval consisting of
// potentially both hole_val and not hol_val elements between bounding elements
// of not hole_val. For example consider s = 1010111, the entire bit vector s is a
// gap_with_holes but 101 is the only gap in s.
{
    max_gap_with_holes_idx_ranges = max_gap_with_holes(vals, max_hole_len, hole_val);
    if(max_gap_with_holes_idx_ranges.empty())
    {
        max_gap_with_holes_len = 0U;
    }
    else
    {
        max_gap_with_holes_len = len_gap(max_gap_with_holes_idx_ranges.front());
    }
    return max_gap_with_holes_len > 0U;
}


std::pair<size_t,size_t> cal_pixel_Lidar_pos(float x_min, float y_min, float cell_size){
    std::pair<size_t, size_t> pixel_Lidar(0, 0);
    std::pair<float, float> ref_Lidar_pos(0.0f, 0.0f);
    float x_diff = ref_Lidar_pos.first - x_min;
    float y_diff = ref_Lidar_pos.second - y_min;
    cout << "x_diff,y_diff " << x_diff << " " << y_diff << endl;
    assert (x_diff > 0.0f && y_diff > 0.0f);
    pixel_Lidar.first = static_cast<size_t>((ref_Lidar_pos.first - x_min) / cell_size);
    pixel_Lidar.second = static_cast<size_t>((ref_Lidar_pos.second - y_min) / cell_size);
    return pixel_Lidar;
}

std::vector<pair<float, float>> organize_edge_candidate(std::vector<Vec3f> &edge_candidates, float edge_thres){
    // organize the candidate vector in increasing order wrt the rho.
    sort(edge_candidates.begin(), edge_candidates.end(), [](const Vec3f &a, const Vec3f &b) { return a[0] < b[0]; });
    std::vector<pair<float, float>> organized_candidates;
//    for (size_t i{0}; i < edge_candidates.size(); i++) {
//        cout << edge_candidates[i][0]<<" "<<edge_candidates[i][1] << endl;
//    }
//    cout<<"----------"<<endl;

//    for(const auto& itr:edge_candidates){
        for(size_t i{0}; i < edge_candidates.size(); i++){
        static float tmp_rho{0.0f};
        static float tmp_theta{0.0f};
        static int cnt{0};
        static bool flag_first{true};
//        cout<<tmp_rho<<" "<<tmp_theta<<" "<<cnt<<" "<<flag_first<<endl;
        if(abs(tmp_rho-edge_candidates[i][0])<edge_thres || flag_first){
            auto ratio = static_cast<float>(cnt) / static_cast<float>(cnt + 1);
//            cout << "ratio: " << ratio << endl;
            if(!flag_first) {
                tmp_rho = tmp_rho * ratio + edge_candidates[i][0] * (1.0f - ratio);
//                cout << "check non first " << tmp_rho << " " << ratio << " " <<edge_candidates[i][0] << endl;
                tmp_theta = tmp_theta * ratio + edge_candidates[i][1] * (1.0f - ratio);
                cnt++;
            }else{
                tmp_rho = edge_candidates[i][0];
                tmp_theta = edge_candidates[i][1];
                cnt = 1;
                flag_first = false;
            }
            if(i == edge_candidates.size()-1) {
                organized_candidates.emplace_back(std::pair<float,float>(tmp_rho,tmp_theta));
                tmp_rho = tmp_theta = 0;
                cnt = 0;
                flag_first = true;
            }
        }else{
            organized_candidates.emplace_back(std::pair<float,float>(tmp_rho,tmp_theta));
            tmp_rho = edge_candidates[i][0];
            tmp_theta = edge_candidates[i][1];
            cnt = 1;
//            flag_first = true;
        }
//        cout<<tmp_rho<<endl;
    }
    cout << "organized: length " << organized_candidates.size() << endl;
    for(auto i : organized_candidates){
        cout << "organized: " << i.first << " " << i.second << endl;
    }

    return organized_candidates;
}

std::pair<size_t,size_t> two_near_edge(size_t tgt_pixel, const vector<pair<float, float>>& all_edges){
    std::pair<size_t, size_t> two_edges(0, all_edges.size());
    for(size_t i{0}; i<all_edges.size(); i++) {
        if (all_edges[i].first < tgt_pixel && i >= two_edges.first) {
            two_edges.first = i;
        }

//        cout<<"check 2 pos: "<<all_edges[i].second<<" "<<tgt_pixel<<" "<<i<<" "<<two_edges.second<<endl;
        if(all_edges[i].first > tgt_pixel && i <= two_edges.second){
            two_edges.second = i;
        }
    }
//    cout << "two_edges: 1: " << two_edges.first << " " << all_edges[two_edges.first].first << endl;
//    cout << "two_edges: 2: " << two_edges.second << " " << all_edges[two_edges.second].first<< endl;

    return two_edges;
}

int main() {

    auto start = chrono::steady_clock::now();

//    std::ifstream ifs("../../Data_folder/output_mtx_20210114211548.txt", std::ifstream::in);
    std::ifstream ifs("../../Data_folder/output_mtx_dig.txt", std::ifstream::in);
    float x_range = 60.0; // hard-coded for this dataset
    float y_range = 80.0; // hard-coded for this dataset
    float z_range = 45.0; // hard-coded for this dataset
    float x_min = -20.0f;
    float y_min = -40.0f;
    float z_min = 15.0;// hard-coded for this dataset
    float cell_size = 0.1; // hard-coded for this dataset
    auto pixel_Lidar = cal_pixel_Lidar_pos(x_min, y_min, cell_size);
    cout << "pixel_Lidar: " << pixel_Lidar.first << " " << pixel_Lidar.second << endl;
    size_t x_num = x_range / cell_size;
    size_t y_num = y_range / cell_size;
    size_t z_num = z_range / cell_size;
    cout << "size: x " << x_num << " ,y " << y_num << " ,z " << z_num << endl;

    float gap_tolerance = 2.0; //meter
    float min_thres = 0.5; //meter
    float max_thres = 4.0; //meter
    float time_factor = 1e6;

    std::string str;
    Eigen::MatrixXf grid(x_num * z_num, y_num);
    grid.setZero();
    int cnt_line = 0;
    auto t_elapsed = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();
    cout << "Time: Start reading matrix at: " << float(t_elapsed) / time_factor << endl;
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

    /** After this line is the same for SHU and this demo*/

    /** Remove points far away from the deck*/
    float margin = 5.0; // meters
    auto deck_range = approx_deck_Z_range(grid, margin, x_num, y_num, z_num, z_min, cell_size);
    cout << "Find proper range for deck: " << deck_range.first << " " << deck_range.second << "\n";
    int z_up_idx = ceil((deck_range.first - z_min) / cell_size);
    int z_bot_idx = floor((deck_range.second - z_min) / cell_size);
    Eigen::MatrixXf grid_deck_cand = grid.block(z_up_idx * x_num, 0, (z_bot_idx - z_up_idx) * x_num, y_num);
    cout << "z_up and z_bot indeices: " << z_up_idx << " " << z_bot_idx << endl;
    cout << "grid_deck_cand size: " << grid_deck_cand.rows() << " " << grid_deck_cand.cols() << endl;

    /** convert eigen matrix to Mat*/
    cout << "Restart counting time." << endl;
    auto t_read = chrono::steady_clock::now();
    cout << "grid sum; " << grid.sum() << endl;
    Eigen::MatrixXf pc_mtx(x_num, y_num);
    pc_mtx.setZero();
    std::vector<bool> z_array(z_num); // for temporary saving z values at specific x,y.
    std::vector<bool> z_array_deck(z_bot_idx - z_up_idx + 1); // for temporary saving z values at specific x,y.
    cout << "z_array_deck size: " << z_array_deck.size() << endl;
    size_t max_hole_len = gap_tolerance / cell_size;
    cout << "max_hole_len: " << max_hole_len << "\n";
    size_t cnt_tmp = 0;
    for (size_t j{1}; j < y_num - 1; j++) // ignore x-y on the edge of the grid
    {
        for (size_t i{1}; i < x_num - 1; i++) // ignore x-y on the edge of the grid
        {
//            cout << "i,j " << i << " " << j << "\n";
            cnt_tmp++;
//            /** Checking Max - Min*/
//            get_z_vector(grid, vec, i, j, x_num);
//            float max = -numeric_limits<float>::infinity();
//            float min = numeric_limits<float>::infinity();
//            for (size_t i_p{0}; i_p < vec.size(); i_p++) {
//                if (vec[i_p] > max && vec[i_p] != 0) max = vec[i_p];
//                if (vec[i_p] < min && vec[i_p] != 0) min = vec[i_p];
//            }
//            pc_mtx(i, j) = max - min;

            /** Find longest consecutive segment*/
//            get_z_std_vector(grid, z_array, i, j, x_num);
            get_z_std_vector(grid_deck_cand, z_array_deck, i, j, x_num);

            size_t max_len{0};
            static thread_local std::vector<std::pair<std::size_t, std::size_t>> gaps;
            gaps.clear();
//            bool is_v_edge = get_max_gap_with_holes(z_array, max_len, gaps, max_hole_len, false);
            bool is_v_edge = get_max_gap_with_holes(z_array_deck, max_len, gaps, max_hole_len, false);

            if (is_v_edge) {
                if (max_len > min_thres / cell_size && max_len < max_thres / cell_size) {
                    pc_mtx(i, j) = max_len;
                }
            }

        }
    }
    cout << "cnt_tmp: " << cnt_tmp << endl;
    t_elapsed = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - t_read).count();
    cout << "Time: Done finding longest segment at: " << float(t_elapsed) / time_factor << endl;
    cout << "pc_mtx sum: " << pc_mtx.sum() << endl;

    /** Converting matrix to image */
//    pc_mtx.setRandom();
//    cout << "test-Mtx: " << pc_mtx << endl;
    cv::Mat_<float> pc_image(x_num, y_num); // create cv mat object

    eigen2cv(pc_mtx, pc_image); // convert eigen matrix to opencv mat
//    pc_image.convertTo(pc_image,CV_8U);
    cout << "pc_image type: " << pc_image.depth() << " " << pc_image.channels() << endl;

    Mat pc_iamge_grey;
    pc_image.convertTo(pc_iamge_grey, CV_8U);

    cout << "Mat: " << pc_image.rows << " " << pc_image.cols << endl;
    namedWindow("Display Image", WINDOW_NORMAL);
    imshow("Display Image", pc_image);
    moveWindow("Display Image", 0, 20);

//    Mat image_edges;
    Mat image_BW(pc_iamge_grey.rows, pc_iamge_grey.cols, CV_8U);
    int lowThreshold = 0;
    const int ratio = 3;
    const int kernel_size = 3;

    t_elapsed = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - t_read).count();
    cout << "Time: Done image conversion at: " << float(t_elapsed) / time_factor << endl;
    int n_rows = pc_iamge_grey.rows;
    int n_cols = pc_iamge_grey.cols;
    cout << "Mat size: " << n_rows << " " << n_cols << endl;
    size_t h_diff_thres = 5; //pixels
    for (size_t i{0}; i < n_rows; i++) {
        for (size_t j{0}; j < n_cols; j++) {
            if (pc_iamge_grey.at<uint8_t>(i, j) >= h_diff_thres) {
                image_BW.at<uint8_t>(i, j) = 1;
            } else {
                image_BW.at<uint8_t>(i, j) = 0;
            }
        }
    }

//    /** Edge detection */
//    Canny(pc_iamge_grey, image_edges, lowThreshold, lowThreshold * ratio, kernel_size);
//    namedWindow("image_edges Image", WINDOW_NORMAL);
//    imshow("image_edges Image", image_edges);
//    moveWindow("image_edges Image", 900, 20);

    vector<Vec3f> HT_v_lines;
    vector<Vec3f> HT_h_lines; // will hold the results of the detection
    cout << "num lines: " << HT_v_lines.size() << endl;

    float HT_theta_resolution = 0.1 * CV_PI / 180; // radian
    float HT_rho_resolution = 0.5f * cell_size; // pixel
    float min_edge_len = 10.0; // meter
    int HT_min_thres_x = int(min_edge_len / cell_size);
    int HT_min_thres_y = int(min_edge_len / cell_size);
    float HT_ang_thres = 1.0f;
    Mat image_color = image_BW * 255;
    cvtColor(image_color, image_color, COLOR_GRAY2BGR);

    /** Find vertical lines*/
    HoughLines(image_BW, HT_v_lines, HT_rho_resolution, HT_theta_resolution, HT_min_thres_x, 0, 0,
               CV_PI / 180 * 1 - HT_ang_thres,
               CV_PI / 180 * 1 + HT_ang_thres); // runs the actual detection
    cout << "num lines: " << HT_v_lines.size() << endl;
    for (size_t i{0}; i < HT_v_lines.size(); i++) {
        cout << "v_line " << i << ": rho " << HT_v_lines[i][0] << ", theta " << HT_v_lines[i][1] / CV_PI * 180.0
             << ", cnt "
             << HT_v_lines[i][2] << endl;
    }
    auto y_lines_organized = organize_edge_candidate(HT_v_lines, 0.5f/cell_size);
    std::pair<size_t,size_t> y_edges = two_near_edge(pixel_Lidar.second, y_lines_organized);
    float yn = (y_lines_organized[y_edges.first].first - pixel_Lidar.second)  *cell_size;
    float yp = (y_lines_organized[y_edges.second].first - pixel_Lidar.second) *cell_size;
    cout << "yn, yp: " << yn << " " << yp << endl;


//    HoughLinesP(image_edges, HT_v_lines, HT_rho_resolution, HT_theta_resolution, HT_min_thres_x,HT_min_thres_x,3);
//    for( size_t i = 0; i < HT_v_lines.size(); i++ )
//    {
//        line( image_color, Point(HT_v_lines[i][0], HT_v_lines[i][1]),
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
        line(image_color, pt1, pt2, Scalar(255, 0, 0), 1, LINE_AA);
    }
    t_elapsed = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - t_read).count();
    cout << "Time: Done finding vertical lines at: " << float(t_elapsed) / time_factor << endl;

    /** Find horizontal lines*/
    HoughLines(image_BW, HT_h_lines, HT_rho_resolution, HT_theta_resolution, HT_min_thres_y, 0, 0,
               CV_PI / 180 * 90 - HT_ang_thres,
               CV_PI / 180 * 90 + HT_ang_thres); // runs the actual detection
    for (size_t i{0}; i < HT_h_lines.size(); i++) {
        cout << "h_line " << i << ": rho " << HT_h_lines[i][0] << ", theta " << HT_h_lines[i][1] / CV_PI * 180.0
             << ", cnt "
             << HT_h_lines[i][2] << endl;
    }
    auto x_lines_organized = organize_edge_candidate(HT_h_lines, 0.5f/cell_size);
    std::pair<size_t,size_t> x_edges = two_near_edge(pixel_Lidar.first, x_lines_organized);
    float xn = (x_lines_organized[x_edges.first].first- pixel_Lidar.first)*cell_size;
    float xp = (x_lines_organized[x_edges.second].first- pixel_Lidar.first)*cell_size;
    cout << "xn, xp: " << xn << " " << xp << endl;


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
        line(image_color, pt1, pt2, Scalar(0, 0, 255), 1, LINE_AA);
    }
    t_elapsed = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - t_read).count();
    cout << "Time: Done finding horizontal lines at: " << float(t_elapsed) / time_factor << endl;

    namedWindow("Detected Lines  - Standard Hough Line Transform", WINDOW_NORMAL);
    Point Lp_1(pixel_Lidar.second - 10, pixel_Lidar.first), Lp_2(pixel_Lidar.second + 10, pixel_Lidar.first), Lp_3(
            pixel_Lidar.second, pixel_Lidar.first - 10), Lp_4(pixel_Lidar.second, pixel_Lidar.first + 10);

    line(image_color, Lp_1, Lp_2, Scalar(0, 0, 255), 1, LINE_AA);
    line(image_color, Lp_3, Lp_4, Scalar(0, 0, 255), 1, LINE_AA);
    imshow("Detected Lines  - Standard Hough Line Transform", image_color);
    moveWindow("Detected Lines  - Standard Hough Line Transform", 900, 20);

    waitKey(0);
    return 0;

}


