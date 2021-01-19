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
//    cout << "get_z_vector: " << step << endl;
    for (size_t i{0}; i < vec.size(); i++) {
        vec[i] = abs(grid(x_idx + i * step, y_idx)) > FLT_EPSILON; // TODO: Checking if this cell filled. The grid matrix will not save float value in the future.
    }
}



#include <vector>
#include <utility>
#include <iostream>

std::pair<std::size_t, std::size_t>
get_bounded_interval(std::vector<bool> const & vals,
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
    Eigen::MatrixXf pc_mtx(x_num, y_num);
    pc_mtx.setZero();
//    Eigen::VectorXf vec(z_num);
    std::vector<bool> z_array(z_num); // for temporary saving z values at specific x,y.
//    cout << "vec size: " << vec.size() << endl;
    for (size_t i{0}; i < x_num; i++) {
        for (size_t j{0}; j < y_num; j++) {
            /** Checking Max - Min*/
//            get_z_vector(grid, vec, i, j, x_num);
//            float max = -numeric_limits<float>::infinity();
//            float min = numeric_limits<float>::infinity();
//            for (size_t i_p{0}; i_p < vec.size(); i_p++) {
//                if (vec[i_p] > max && vec[i_p] != 0) max = vec[i_p];
//                if (vec[i_p] < min && vec[i_p] != 0) min = vec[i_p];
//            }
//            pc_mtx(i, j) = max - min;

            /** Find longest consecutive segment*/
            get_z_std_vector(grid, z_array, i, j, x_num);
            size_t max_len{0};
            std::vector<std::pair<std::size_t, std::size_t>> gaps;
            bool is_v_edge = get_max_gap_with_holes(z_array,  max_len, gaps, 10, 0);
            //TODO-Continue: Continue here on selecting longest segment
        }
    }


    /** Converting matrix to image */
//    pc_mtx.setRandom();
    cout << "test-Mtx: " << pc_mtx << endl;
    cv::Mat_<float> pc_image(x_num, y_num); // create cv mat object

    eigen2cv(pc_mtx, pc_image); // convert eigen matrix to opencv mat
//    pc_image.convertTo(pc_image,CV_8U);
    cout << "pc_image type: " << pc_image.depth() << " " << pc_image.channels() << endl;

    Mat pc_iamge_grey;
    pc_image.convertTo(pc_iamge_grey, CV_8U);

    cout << "Mat: " << pc_image.rows << " " << pc_image.cols << endl;
    namedWindow("Display Image", WINDOW_NORMAL);
    imshow("Display Image", pc_image);
    moveWindow("Display Image", 200,20);

    Mat image_edges;
    int lowThreshold = 0;
    const int ratio = 3;
    const int kernel_size = 3;


    /** Edge detection */
    Canny(pc_iamge_grey, image_edges, lowThreshold, lowThreshold * ratio, kernel_size);
    namedWindow("image_edges Image", WINDOW_NORMAL);
    imshow("image_edges Image", image_edges);
    moveWindow("image_edges Image", 600,20);

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
    moveWindow("Detected Lines  - Standard Hough Line Transform", 1000,20);

    waitKey(0);
    return 0;

}


