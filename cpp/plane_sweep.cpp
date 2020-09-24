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
 * @param dtu - Flag specifying whether or not the data is a DTU dataset
 *
 */
Mat plane_sweep(vector<float> &cost_volume, const vector<Mat> &images, const vector<Mat> &K, const vector<Mat> &R, const vector<Mat> &t, const vector<Mat> &bounds, int index, int depth_count, int window_size, bool dtu) {

    Size shape = images[index].size();
    float cost;
    int img_count = images.size();
    int offset = (window_size-1)/2;

    vector<vector<Mat>> homogs;
    Mat depth_map = Mat::zeros(shape, CV_32F);
    Mat depth_values(shape, CV_32F);
    fill(cost_volume.begin(), cost_volume.end(), numeric_limits<float>::max());

    // create plane norm
    Mat n = Mat::zeros(3,1,CV_32F);
    n.at<float>(2,0) = 1;

    // compute bounds
    float min_dist;
    float max_dist;

    // check to see if its a dtu dataset
    if (dtu) {
        min_dist = bounds[index].at<float>(0,0);
        max_dist = bounds[index].at<float>(1,0);
    } else {
        Mat n0 = Mat::zeros(3,1,CV_32F);
        n0.at<float>(2,0) = 1;
        Mat n1 = R[index]*n0;

        Mat v_min = (bounds[index].row(1)).t() - t[index];
        Mat v_max = (bounds[index].row(0)).t() - t[index];

        min_dist = v_min.dot(n1);
        max_dist = v_max.dot(n1);
    }

    float interval = (max_dist-min_dist)/depth_count;

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
            Mat M1;
            // check to see if its a dtu dataset
            if (dtu) {
                M1.push_back(R[index].t());
                M1.push_back(t[index].t());
            } else {
                M1.push_back(R[index]);
                M1.push_back((-R[index].t()*t[index]).t());
            }
            M1 = M1.t();
            Mat temp1;
            temp1.push_back(Mat::zeros(3,1,CV_32F));
            temp1.push_back(Mat::ones(1,1,CV_32F));
            M1.push_back(temp1.t());

            Mat M2;
            // check to see if its a dtu dataset
            if (dtu) {
                M2.push_back(R[i].t());
                M2.push_back(t[i].t());
            } else {
                M2.push_back(R[i]);
                M2.push_back((-R[i].t()*t[i]).t());
            }
            M2 = M2.t();
            Mat temp2;
            temp2.push_back(Mat::zeros(3,1,CV_32F));
            temp2.push_back(Mat::ones(1,1,CV_32F));
            M2.push_back(temp2.t());

            Mat M = M2*M1.inv();

            Mat R_rel = M(Rect(0,0,3,3));
            Mat t_rel = M(Rect(3,0,1,3));

            // compute homography
            Mat H = K[i]*(R_rel + (t_rel * n.t()/z_curr)) * K[index].inv();

            img_homogs.push_back(H);
        }
        homogs.push_back(img_homogs);
        z_curr+=interval;
    }
//} //omp
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

                    //cout << "d: " << d << "\nind: " << ind << endl;
                    //cout << "homogs[d][ind]: " << homogs[d][ind] << endl << endl;
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

                z_curr+=interval;
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
            
            depth_values.at<float>(r,c) = min_dist + (interval*best_depth);
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

/*
 * @brief Performs depth map fusion using the stability-based notion of a depth estimate
 *
 * @param depth_maps - The container holding the depth maps to be fused
 * @param conf_maps - The container holding the confidence maps needed for the fusion process
 *
 */
void stability_fusion(const vector<Mat> &depth_maps, const vector<Mat> &conf_maps){
    cout << "Running stability-based fusion..." << endl;
}

/*
 * @brief Performs depth map fusion using the confidence-based notion of a depth estimate
 *
 * @param depth_maps - The container holding the depth maps to be fused
 * @param conf_maps - The container holding the confidence maps needed for the fusion process
 *
 */
void confidence_fusion(const vector<Mat> &depth_maps, const vector<Mat> &conf_maps, const vector<Mat> &K, const vector<Mat> &R, const vector<Mat> &t, const vector<Mat> &bounds, const int index, const int depth_count, const Size shape, const bool dtu, int window_size, int scale){
    int depth_map_count = depth_maps.size();

    // compute bounds
    float min_dist;
    float max_dist;

    // check to see if its a dtu dataset
    if (dtu) {
        min_dist = bounds[index].at<float>(0,0);
        max_dist = bounds[index].at<float>(1,0);
    } else {
        Mat n0 = Mat::zeros(3,1,CV_32F);
        n0.at<float>(2,0) = 1;
        Mat n1 = R[index]*n0;

        Mat v_min = (bounds[index].row(1)).t() - t[index];
        Mat v_max = (bounds[index].row(0)).t() - t[index];

        min_dist = v_min.dot(n1);
        max_dist = v_max.dot(n1);
    }

    vector<Mat> P;
    vector<Mat> K_fr;

    cout << "\tPre-Computing Intrinsics/Extrinsics..." << endl;
    // pre-compute intrinsics/extrinsics
    for (int i=0; i<depth_map_count; ++i) {
        Mat P_i;
        // check to see if its a dtu dataset
        if (dtu) {
            P_i.push_back(R[i].t());
            P_i.push_back(t[i].t());
        } else {
            P_i.push_back(R[i]);
            P_i.push_back((-R[i].t()*t[i]).t());
        }
        P_i = P_i.t();
        Mat temp1;
        temp1.push_back(Mat::zeros(3,1,CV_32F));
        temp1.push_back(Mat::ones(1,1,CV_32F));
        P_i.push_back(temp1.t());

        P.push_back(P_i);

        Mat K_i;
        K_i.push_back(K[i].t());

        Mat temp2;
        temp2.push_back(Mat::zeros(1,3,CV_32F));
        K_i.push_back(temp2);
        K_i = K_i.t();

        Mat temp3;
        temp3.push_back(Mat::zeros(3,1,CV_32F));
        temp3.push_back(Mat::ones(1,1,CV_32F));
        K_i.push_back(temp3.t());

        K_fr.push_back(K_i);
    }

    cout << "\tRendering depth maps into reference view..." << endl;
    vector<Mat> d_refs;
    vector<Mat> c_refs;
    const int rows = shape.height;
    const int cols = shape.width;

    // TEST
    Mat t_world_l = Mat::zeros(4,1,CV_32F);
    t_world_l.at<float>(0,0) = 42.83;
    t_world_l.at<float>(1,0) = -11.35;
    t_world_l.at<float>(2,0) = 652.64;
    t_world_l.at<float>(3,0) = 1;

    Mat t_world_r = Mat::zeros(4,1,CV_32F);
    t_world_l.at<float>(0,0) = 28.73;
    t_world_l.at<float>(1,0) = -52.2;
    t_world_l.at<float>(2,0) = 665.978;
    t_world_l.at<float>(3,0) = 1;

    cout << P[index] << endl;
    cout << K_fr[index] << endl;
    
    Mat t_p_l = K_fr[index] * P[index] * t_world_l;

    t_p_l.at<float>(0,0) = t_p_l.at<float>(0,0)/t_p_l.at<float>(2,0);
    t_p_l.at<float>(1,0) = t_p_l.at<float>(1,0)/t_p_l.at<float>(2,0);
    t_p_l.at<float>(2,0) = t_p_l.at<float>(2,0)/t_p_l.at<float>(2,0);

    int c_t_l = (int) floor(t_p_l.at<float>(0,0));
    int r_t_l = (int) floor(t_p_l.at<float>(1,0));
    cout << c_t_l << "," << r_t_l << endl << endl;

    Mat t_p_r = K_fr[index] * P[index] * t_world_r;

    t_p_r.at<float>(0,0) = t_p_r.at<float>(0,0)/t_p_r.at<float>(2,0);
    t_p_r.at<float>(1,0) = t_p_r.at<float>(1,0)/t_p_r.at<float>(2,0);
    t_p_r.at<float>(2,0) = t_p_r.at<float>(2,0)/t_p_r.at<float>(2,0);

    int c_t_r = (int) floor(t_p_r.at<float>(0,0));
    int r_t_r = (int) floor(t_p_r.at<float>(1,0));
    cout << c_t_r << "," << r_t_r << endl << endl;
    exit(0);


    // TEST







    for (int d=0; d < depth_map_count; ++d) {
        if (d==index) {
            d_refs.push_back(depth_maps[index]);
            c_refs.push_back(conf_maps[index]);

            // write ply file
            vector<int> green = {0, 255, 0};
            write_ply(depth_maps[index], K_fr[index], P[index], "test_ref_" + to_string(index+1) + ".ply", green);

            continue;
        }

        Mat d_ref = Mat::zeros(shape, CV_32F);
        Mat c_ref = Mat::zeros(shape, CV_32F);

#pragma omp parallel num_threads(24)
{
        #pragma omp for collapse(2) 
        for (int r=0; r<rows; ++r) {
            for (int c=0; c<cols; ++c) {
                float depth = depth_maps[d].at<float>(r,c);
                float conf = conf_maps[d].at<float>(r,c);

                // compute corresponding (x,y) locations
                Mat x_1(4,1,CV_32F);
                x_1.at<float>(0,0) = c;
                x_1.at<float>(1,0) = r;
                x_1.at<float>(2,0) = 1;
                x_1.at<float>(3,0) = 1/depth;

                // find 3D world coord of back projection
                Mat cam_coords = K_fr[d].inv() * x_1;
                Mat X_world = P[d].inv() * cam_coords;
                X_world.at<float>(0,0) = X_world.at<float>(0,0) / X_world.at<float>(0,3);
                X_world.at<float>(0,1) = X_world.at<float>(0,1) / X_world.at<float>(0,3);
                X_world.at<float>(0,2) = X_world.at<float>(0,2) / X_world.at<float>(0,3);
                X_world.at<float>(0,3) = X_world.at<float>(0,3) / X_world.at<float>(0,3);


                // find pixel location in reference image
                Mat x_2 = K_fr[index] * P[index] * X_world;

                x_2.at<float>(0,0) = x_2.at<float>(0,0)/x_2.at<float>(2,0);
                x_2.at<float>(1,0) = x_2.at<float>(1,0)/x_2.at<float>(2,0);
                x_2.at<float>(2,0) = x_2.at<float>(2,0)/x_2.at<float>(2,0);

                int c_p = (int) floor(x_2.at<float>(0,0));
                int r_p = (int) floor(x_2.at<float>(1,0));

                if (c_p < 0 || c_p >= shape.width || r_p < 0 || r_p >= shape.height) {
                    continue;
                }

                d_ref.at<float>(r_p,c_p) = depth;
                c_ref.at<float>(r_p,c_p) = conf;
            }
        }
} //omp parallel

        d_refs.push_back(d_ref);
        c_refs.push_back(c_ref);
    }
    
    // Fuse depth maps
    float f;
    float initial_f;
    float C;
    float eps = 1.5;

    vector<Mat>::const_iterator d_map;
    vector<Mat>::const_iterator c_map;
    Mat fused_map = Mat::zeros(shape,CV_32F);
    Mat fused_conf = Mat::zeros(shape,CV_32F);

    cout << "\tFusing depth maps..." << endl;
    for (int r=0; r<rows; ++r) {
        for (int c=0; c<cols; ++c) {
            // set initial depth estimate and confidence value
            f = 0.0;
            initial_f = 0.0;
            C = 0.0;

            int initial_d = 0;

            for (int d=0; d < depth_map_count; ++d) {
                if (c_refs[d].at<float>(r,c) > C) {
                    f = d_refs[d].at<float>(r,c);
                    C = c_refs[d].at<float>(r,c);
                    initial_d = d;
                }
            }

            // store the initial depth value
            initial_f = f;

            // for each depth map:
            for (int d=0; d < depth_map_count; ++d) {
                // skip checking for the most confident depth map
                if (d == initial_d) {
                    continue;
                }

                float curr_depth = d_refs[d].at<float>(r,c);
                float curr_conf = c_refs[d].at<float>(r,c);

                // if depth is close to initial depth:
                if (abs(curr_depth - initial_f) < eps) {
                    if((C + curr_conf) != 0) {
                        f = (f*C + curr_depth*curr_conf) / (C + curr_conf);
                    }
                    C += curr_conf;
                } 
                // if depth is too close (occlusion):
                else if(curr_depth < initial_f) {
                    C -= curr_conf;
                }
                // if depth is too large (free space violation):
                else if(curr_depth > initial_f) {
                    //C -= c_map->at<float>(r,c);
                    // C -= C_i(P(X))
                    // compute corresponding (x,y) locations
                    Mat x_1(3,1,CV_32F);
                    x_1.at<float>(0,0) = c;
                    x_1.at<float>(1,0) = r;
                    x_1.at<float>(2,0) = 1;
                    x_1.at<float>(3,0) = 1/initial_f;

                    // find 3D world coord of back projection
                    Mat cam_coords = K_fr[initial_d].inv() * x_1;
                    Mat X_world = P[d].inv() * cam_coords;
                    X_world.at<float>(0,0) = X_world.at<float>(0,0) / X_world.at<float>(0,3);
                    X_world.at<float>(0,1) = X_world.at<float>(0,1) / X_world.at<float>(0,3);
                    X_world.at<float>(0,2) = X_world.at<float>(0,2) / X_world.at<float>(0,3);
                    X_world.at<float>(0,3) = X_world.at<float>(0,3) / X_world.at<float>(0,3);

                    // find pixel location in reference image
                    Mat x_2 = K_fr[d] * P[d] * X_world;

                    x_2.at<float>(0,0) = x_2.at<float>(0,0)/x_2.at<float>(2,0);
                    x_2.at<float>(1,0) = x_2.at<float>(1,0)/x_2.at<float>(2,0);
                    x_2.at<float>(2,0) = x_2.at<float>(2,0)/x_2.at<float>(2,0);

                    int c_p = (int) floor(x_2.at<float>(0,0));
                    int r_p = (int) floor(x_2.at<float>(1,0));

                    if (c_p < 0 || c_p >= shape.width || r_p < 0 || r_p >= shape.height) {
                        continue;
                    }

                    C -= c_refs[d].at<float>(r_p,c_p);
                }
            }

            if (C <= 0.0) {
                f = -1.0;
                C = -1.0;
            }
            fused_map.at<float>(r,c) = f;
            fused_conf.at<float>(r,c) = C;
        }
    }

    int w = 5;
    int w_offset = (w-1)/2;
    int w_inliers = (w*w)/2;

    int w_s = 3;
    int w_s_offset = (w_s-1)/2;
    int w_s_inliers = (w_s*w_s)/2;

    Mat filled_map = Mat::zeros(shape, CV_32F);
    Mat smoothed_map = Mat::zeros(shape, CV_32F);

    // Fill in holes (-1 values) in depth map
    for (int r=w_offset; r<rows-w_offset; ++r) {
        for (int c=w_offset; c<cols-w_offset; ++c) {
            if (fused_map.at<float>(r,c) < 0.0){
                filled_map.at<float>(r,c) = med_filt(fused_map(Rect(c-w_offset,r-w_offset,w,w)), w, w_inliers);
            } else {
                filled_map.at<float>(r,c) = fused_map.at<float>(r,c);
            }
        }
    }

    // Smooth out inliers
    for (int r=w_s_offset; r<rows-w_s_offset; ++r) {
        for (int c=w_s_offset; c<cols-w_s_offset; ++c) {
            if (filled_map.at<float>(r,c) != -1){
                smoothed_map.at<float>(r,c) = med_filt(filled_map(Rect(c-w_s_offset,r-w_s_offset,w_s,w_s)), w_s, w_s_inliers);
            }
        }
    }

    // write ply file
    vector<int> gray = {200, 200, 200};
    write_ply(smoothed_map, K_fr[index], P[index], "test_fused_" + to_string(index+2) + ".ply", gray);

    write_map(smoothed_map, "depth_fused_" + to_string(index) + ".png", scale);
    write_map(fused_conf, "conf_fused_" + to_string(index) + ".png", scale);
}

int main(int argc, char **argv) {
    
    if (argc != 6) {
        fprintf(stderr, "Error: usage %s <0 if camera centers / 1 if translation vectors> <path-to-images> <down-sample-scale> <depth-increments> <sad-window-size>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    bool dtu = atoi(argv[1]);
    char *data_path = argv[2];
    float scale = atof(argv[3]);
    int depth_count = atoi(argv[4]);
    int window_size = atoi(argv[5]);
    size_t str_len = strlen(data_path);

    if (data_path[str_len-1] != '/') {
        strcat(data_path, "/");
    }
    
    vector<Mat> images;
    vector<Mat> depth_maps;
    vector<Mat> K;
    vector<Mat> R;
    vector<Mat> t;
    vector<Mat> bounds;
    vector<Mat> confidence_maps;
    Size shape;
    
    // load images, K's, R's, t's, bounds
    printf("Loading data...\n");
    load_data(&images, &K, &R, &t, &bounds, data_path, dtu);

    down_sample(&images, &K, scale);

    int img_count = images.size();
    int offset = 1;
    int end_offset = img_count-offset;
    //int end_offset = 8;

    // build depth map
    for (int i=offset; i<end_offset; ++i) {
        printf("Computing depth map for image %d/%d...\n",(i+1)-offset,end_offset-offset);

        shape = images[i].size();
        vector<float> cost_volume(shape.width*shape.height*depth_count);

        Mat depth_map = plane_sweep(cost_volume, images, K, R, t, bounds, i, depth_count, window_size, dtu);

        Mat conf_map = Mat::zeros(shape, CV_32F);
        build_conf_map(depth_map, conf_map, cost_volume, shape, depth_count);

        confidence_maps.push_back(conf_map);
        depth_maps.push_back(depth_map);

        // write depth image
        write_map(depth_map, "depth_" + to_string(i) + ".png", scale);

        // write confidence image
        write_map(conf_map, "conf_" + to_string(i) + ".png", scale);
    }

    int depth_map_count = depth_maps.size();
    int fusion_offset = 1;

    //stability_fusion(depth_maps, confidence_maps);
    for (int i=fusion_offset; i<depth_map_count-fusion_offset; ++i) {
        printf("Running confidence-based fusion for depth map %d/%d...\n",(i+1)-fusion_offset,depth_map_count-(2*fusion_offset));
        vector<Mat> offset_K(K.begin()+offset, K.begin()+end_offset);
        vector<Mat> offset_R(R.begin()+offset, R.begin()+end_offset);
        vector<Mat> offset_t(t.begin()+offset, t.begin()+end_offset);
        vector<Mat> offset_bounds(bounds.begin()+offset, bounds.begin()+end_offset);

        confidence_fusion(depth_maps, confidence_maps, offset_K, offset_R, offset_t, offset_bounds, i, depth_count, shape, dtu, window_size, scale);
    }

    return EXIT_SUCCESS;
}
