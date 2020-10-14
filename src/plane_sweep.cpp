#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <dirent.h>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <fstream>

#include "plane_sweep.h"
#include "util.h"

using namespace cv;
using namespace std;


/*
 * @brief Performs the Plane Sweep stereo algorithm for the given reference and target images
 *
 * @param cost_volume - The container to be populated with the cost volume for the reference image
 * @param images - The container holding the images
 * @param K - The container holding the intrinsic matrices for the images
 * @param R - The container holding the rotation matrices for the images
 * @param t - The container holding the translation vectors for the images
 * @param P - The container holding the projection matrices for the images
 * @param bounds - The container holding the bounds information for the images
 * @param index - The reference image index
 * @param depth_count - The number of depth increments to be used in plane sweep
 * @param window_size - The window size used for SAD (window_size x window_size)
 *
 */
Mat plane_sweep(vector<float> &cost_volume, const vector<Mat> &images, const vector<Mat> &K, const vector<Mat> &P, const Bounds &bounds, int index, int depth_count, int window_size) {

    Size shape = images[index].size();
    float cost;
    int img_count = images.size();
    int offset = (window_size-1)/2;

    vector<vector<Mat>> homogs;
    Mat depth_map = Mat::zeros(shape, CV_32F);
    Mat depth_values = Mat::zeros(shape, CV_32F);
    fill(cost_volume.begin(), cost_volume.end(), numeric_limits<float>::max());

    // create plane norm
    Mat n = Mat::zeros(3,1,CV_32F);
    n.at<float>(2,0) = 1;

    // compute bounds
    float min_dist = bounds.min_dist;
    float increment = bounds.increment;

    cout << "\tPre-computing homographies..." << endl;
    // pre-compute homographies
    float z_curr = min_dist;
    for (int d=0; d<depth_count; ++d) {
        vector<Mat> img_homogs;

        for (int i=0; i<img_count; ++i) {
            if (index == i) {
                continue;
            }

            // compute relative extrinsics
            Mat P_rel = P[i]*P[index].inv();

            Mat R_rel = P_rel(Rect(0,0,3,3));
            Mat t_rel = P_rel(Rect(3,0,1,3));

            // compute homography
            Mat H = K[i]*(R_rel + (t_rel * n.t()/z_curr)) * K[index].inv();

            img_homogs.push_back(H);
        }
        homogs.push_back(img_homogs);
        z_curr+=increment;
    }
    int num_matches;

    cout << "\tBuilding depth map..." << endl;
#pragma omp parallel num_threads(24) private(cost,num_matches)
{
    #pragma omp for collapse(2) 
    for (int y=offset; y<(shape.height-offset); ++y) {
        for (int x=offset; x<(shape.width-offset); ++x) {
            float z_curr = min_dist;
            for (int d=0; d<depth_count; ++d) {
                cost = 0;
                num_matches = 0;
                int ind = 0;
                for (int i=0; i<img_count; ++i) {
                    if (index == i) {
                        continue;
                    }
                    
                    // compute corresponding (x,y) locations
                    Mat x_1(3,1,CV_32F);
                    x_1.at<float>(0,0) = x;
                    x_1.at<float>(1,0) = y;
                    x_1.at<float>(2,0) = 1;

                    Mat x_2 = homogs[d][ind]*x_1;
                    x_2.at<float>(0,0) = x_2.at<float>(0,0)/x_2.at<float>(2,0);
                    x_2.at<float>(1,0) = x_2.at<float>(1,0)/x_2.at<float>(2,0);
                    x_2.at<float>(2,0) = x_2.at<float>(2,0)/x_2.at<float>(2,0);

                    int x_p = (int) floor(x_2.at<float>(0,0));
                    int y_p = (int) floor(x_2.at<float>(1,0));

                    if (x_p < offset || x_p >= shape.width-offset || y_p < offset || y_p >= shape.height-offset) {
                        continue;
                    }
                    num_matches++;

                    // compute pizel matching cost
                    // SAD on (window_size x window_size) regions
                    Mat sub1 = images[index](Rect(x-offset,y-offset,window_size,window_size));
                    Mat sub2 = images[i](Rect(x_p-offset,y_p-offset,window_size,window_size));
                    Mat diff;
                    Scalar s;

                    absdiff(sub1,sub2,diff);
                    s = sum(diff);

                    cost += sqrt((s[0]*s[0]) + (s[1]*s[1]) + (s[2]*s[2]));
                    ++ind;
                }

                if(num_matches > 0) {
                    long ind = static_cast<long>(y*shape.width*depth_count) + static_cast<long>(x*depth_count) + static_cast<long>(d);
                    cost_volume[ind] = cost/num_matches;
                }

                z_curr+=increment;
            }
        }
    }
} //omp parallel

    // build depth map
    float min_cost;
    int best_depth;
    long i;

    for (int r=offset; r<shape.height-offset; ++r) {
        for (int c=offset; c<shape.width-offset; ++c) {
            min_cost = numeric_limits<float>::max();
            best_depth = 0;

            for (float d=0; d<depth_count; ++d) {
                i = static_cast<long>(r*shape.width*depth_count) + static_cast<long>(c*depth_count) + static_cast<long>(d);
                if ((cost_volume[i] >= 0) && (cost_volume[i] < min_cost)) {
                    min_cost = cost_volume[i];
                    best_depth = d;
                }
            }
            
            depth_values.at<float>(r,c) = min_dist + (increment*best_depth);
        }
    }

    return depth_values;
}

/*
 * @brief Builds the confidence map for the given reference image based on the estimated depth map and cost volume
 *
 * @param depth_map - The container holding the depth map for a given reference view
 * @param conf_map - The container to be populated with the confidence values for each pixel of a given reference view
 * @param cost_volume - The container holding the cost volume for a given reference view
 * @param shape - The size of the given depth map
 * @param depth_count - The number of depth increments used in plane sweep
 * @param sigma - The standard deviation of the assumed gaussian noice for the depth estimates
 *
 */
void build_conf_map(const Mat &depth_map, Mat &conf_map, const vector<float> &cost_volume, const Size shape, const int depth_count) {
    cout << "\tBuilding confidence map..." << endl;
    int rows = shape.height;
    int cols = shape.width;
    long ind;
    float conf;
    float curr_cost;
    float min1_cost;
    float min2_cost;
    float eps = 1e-3;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            conf = 0.0;
            min1_cost = numeric_limits<float>::max();
            min2_cost = numeric_limits<float>::max();


            for(int d = 0; d < depth_count; ++d) {
                ind = static_cast<long>(r*cols*depth_count) + static_cast<long>(c*depth_count) + static_cast<long>(d);

                curr_cost = cost_volume[ind];

                if (curr_cost < min1_cost) {
                    float temp = min1_cost;
                    min1_cost = curr_cost;
                    min2_cost = temp;
                } else if(curr_cost < min2_cost) {
                    min2_cost = curr_cost;
                }
            }

            conf = 1 - ((min1_cost + eps) / (min2_cost + eps));
            conf_map.at<float>(r,c) = conf;
        }
    }
}

int main(int argc, char **argv) {
    
    if (argc != 5) {
        fprintf(stderr, "Error: usage %s <path-to-images> <down-sample-scale> <depth-increments> <sad-window-size>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    string data_path = argv[1];
    float scale = atof(argv[2]);
    int depth_count = atoi(argv[3]);
    int window_size = atoi(argv[4]);
    size_t str_len = data_path.length();

    if (data_path[str_len-1] != '/') {
        data_path += "/";
    }
    
    vector<Mat> images;
    vector<Mat> depth_maps;
    vector<Mat> K;
    vector<Mat> P;
    Bounds bounds;
    vector<Mat> confidence_maps;
    Size shape;

    // load images, K's, P's, bounds
    printf("Loading data...\n");
    load_images(&images, data_path);
    load_camera_params(&K, &P, &bounds, data_path);

    down_sample(&images, &K, scale);

    int img_count = images.size();

    // build depth map
    for (int i=0; i<img_count; ++i) {
        printf("Computing depth map for image %d/%d...\n",(i+1),img_count);

        shape = images[i].size();
        vector<float> cost_volume(shape.width*shape.height*depth_count);

        Mat depth_map = plane_sweep(cost_volume, images, K, P, bounds, i, depth_count, window_size);

        Mat conf_map = Mat::zeros(shape, CV_32F);
        build_conf_map(depth_map, conf_map, cost_volume, shape, depth_count);

        confidence_maps.push_back(conf_map);
        depth_maps.push_back(depth_map);

        // write depth image
        write_map(depth_map, data_path + "depth_maps/depth_" + to_string(i) + ".csv");
        display_map(depth_map, "disp_map_" + to_string(i) + ".png");

        // write confidence image
        write_map(conf_map, data_path + "conf_maps/conf_" + to_string(i) + ".csv");
        display_map(conf_map, "disp_conf_" + to_string(i) + ".png");
    }
    return EXIT_SUCCESS;
}
