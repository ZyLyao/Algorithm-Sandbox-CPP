#include <Eigen/Dense>
#include <iostream>
#include <vector>

#define MAX_MASK_SIZE 5

using namespace std;
using namespace Eigen;

int num_x = 5;
int num_y = 5;
int num_z = 5;
Eigen::Matrix<bool, Dynamic, Dynamic> grid_of_num(num_x * num_y,num_z);


bool
get_blur_grid(Eigen::Matrix<bool, Dynamic, Dynamic> & grid_blur,
                          const std::vector<size_t> & mask)
{
    size_t dim_blur = mask.size();
    if(dim_blur <= 0 || dim_blur > 3)
    {
        return 0; // return error if the dimension setting is unreasonable.
    }
    else
    {
        size_t mask_x{ 0 }, mask_y{ 0 }, mask_z{ 0 };
        switch(dim_blur)
        {
            case 3:
                mask_z = mask[2];
            case 2:
                mask_y = mask[1];
            case 1:
                mask_x = mask[0];
        }

        /** Exit if size not proper*/
        if(mask_x > MAX_MASK_SIZE || mask_y > MAX_MASK_SIZE || mask_z > MAX_MASK_SIZE)
        {
            cerr << "mask size for get_blur_grid() is too large." << endl;
            return 0;
        }

        /** Start creating blurred grid*/
        int x_quotient, x_remainer, layer_minus, layer_plus;
        size_t row_min, row_max, col_min, col_max;
        for(size_t i{ 0 }; i < grid_of_num.rows(); i++)
        {
            for(size_t j{ 0 }; j < grid_of_num.cols(); j++)
            {
                x_quotient = static_cast<int>(i) / num_x;
                x_remainer = static_cast<int>(i) % num_x;
                size_t x0_this_lasyer = i - x_remainer;
                row_min = std::max(x_remainer - static_cast<int>(mask_x), 0) + x0_this_lasyer;

                row_max = std::min(i + mask_x, x0_this_lasyer + num_x - 1);
                col_min = std::max(static_cast<int>(j) - static_cast<int>(mask_y), 0);
                col_max = std::min(j + mask_y, static_cast<size_t>(num_y - 1));
                layer_minus
                        = x_quotient - static_cast<int>(mask_z) > 0 ? -static_cast<int>(mask_z) : 0;
                layer_plus = x_quotient + static_cast<int>(mask_z) + 1 > num_z
                             ? static_cast<int>(num_z) - x_quotient
                             : static_cast<int>(mask_z) + 1 ;

                if(grid_of_num(i, j) == true)
                {
                    if(mask_z > 0)
                    {
                        for(int layer{ layer_minus }; layer < layer_plus; layer++)
                        {
                            size_t row_start = row_min + num_x * layer;
                            size_t row_num = row_max - row_min + 1;
                            size_t col_num = col_max - col_min + 1;
                            grid_blur.block(row_start, col_min, row_num, col_num).setOnes();
                        }
                    }
                    else
                    {
                        size_t row_start = row_min;
                        size_t row_num = row_max - row_min+ 1;
                        size_t col_num = col_max - col_min+ 1;
                        grid_blur.block(row_start, col_min, row_num, col_num).setOnes();
                    }
                }
            }
        }
        return 1;
    }
}

int main() {
    Eigen::Matrix<bool, Dynamic, Dynamic>  grid_blur(num_x * num_y,num_z);
//    grid_of_num(0*num_x + 0,0) = 1;
//    grid_of_num(0*num_x + 4,0) = 1;
//    grid_of_num(0*num_x + 0,4) = 1;
//    grid_of_num(0*num_x + 4,4) = 1;
//    grid_of_num(4*num_x + 0,0) = 1;
//    grid_of_num(4*num_x + 4,0) = 1;
//    grid_of_num(4*num_x + 0,4) = 1;
//    grid_of_num(4*num_x + 4,4) = 1;
//    grid_of_num(0*num_x + 2,2) = 1;
    grid_of_num(2*num_x + 2,2) = 1;
//    grid_of_num(2*num_x + 1,1) = 1;


    bool flag = get_blur_grid(grid_blur, {1,1,1});
    if(flag){
        cout<<"succeed."<<endl;
    }else{
        cerr<<"failed"<<endl;;
    }
    cout << grid_blur << endl;
    return 0;
}
