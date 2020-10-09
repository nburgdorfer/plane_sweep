#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <dirent.h>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <fstream>

#include "util.h"
#include "depth_fusion.h"

/*
 * @brief Performs depth map fusion using the confidence-based notion of a depth estimate
 *
 * @param depth_maps - The container holding the depth maps to be fused
 * @param conf_maps - The container holding the confidence maps needed for the fusion process
 *
 */
void confidence_fusion(const vector<Mat> &depth_maps, Mat &fused_map, const vector<Mat> &conf_maps, Mat &fused_conf, const vector<Mat> &K, const vector<Mat> &R, const vector<Mat> &t, const vector<Mat> &bounds, const int index, const int scale, const string data_path){
    int depth_map_count = depth_maps.size();
    Size size = depth_maps[0].size();

    // compute bounds
    float min_dist = bounds[index].at<float>(0,0);
    float max_dist = bounds[index].at<float>(1,0);

    vector<Mat> P;
    vector<Mat> K_fr;

    cout << "\tPre-Computing Intrinsics/Extrinsics..." << endl;
    // pre-compute intrinsics/extrinsics
    for (int i=0; i<depth_map_count; ++i) {
        Mat P_i;
        P_i.push_back(R[i].t());
        P_i.push_back(t[i].t());

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
    const int rows = size.height;
    const int cols = size.width;

    for (int d=0; d < depth_map_count; ++d) {
        if (d==index) {
            d_refs.push_back(depth_maps[index]);
            c_refs.push_back(conf_maps[index]);
            continue;
        }

        Mat d_ref = Mat::zeros(size, CV_32F);
        Mat c_ref = Mat::zeros(size, CV_32F);

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
                
                if (c_p < 0 || c_p >= size.width || r_p < 0 || r_p >= size.height) {
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

                    if (c_p < 0 || c_p >= size.width || r_p < 0 || r_p >= size.height) {
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

    Mat filled_map = Mat::zeros(size, CV_32F);
    Mat smoothed_map = Mat::zeros(size, CV_32F);

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
    
    fused_map = smoothed_map;

    // write ply file
    vector<int> gray = {200, 200, 200};
    write_ply(fused_map, K_fr[index], P[index], data_path+"points_mvs/"+to_string(index)+"_nate006_l3.ply", gray);
}


int main(int argc, char **argv) {
    
    if (argc != 3) {
        fprintf(stderr, "Error: usage %s <path-to-depth-maps> <down-sample-scale>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    string data_path = argv[1];
    float scale = atof(argv[2]);
    size_t str_len = data_path.length();

    if (data_path[str_len-1] != '/') {
        data_path += "/";
    }
    
    vector<Mat> depth_maps;
    vector<Mat> conf_maps;
    vector<Mat> K;
    vector<Mat> R;
    vector<Mat> t;
    vector<Mat> bounds;
    
    // load images, K's, R's, t's, bounds
    printf("Loading data...\n");
    load_depth_maps(&depth_maps, data_path);
    load_conf_maps(&conf_maps, data_path);
    load_camera_params(&K, &R, &t, data_path);
    load_bounds(&bounds, data_path);

    down_sample_k(&K, scale);

    int depth_map_count = depth_maps.size();
    Size size = depth_maps[0].size();
    int image_count = K.size();
    int offset=0;
    int end_offset = image_count-offset;
    //int end_offset=8;
    int fusion_offset = 0;
    
    for (int r=0; r<size.height; ++r){
        for(int c=0; c<size.width; ++c) {
            cout << conf_maps[0].at<float>(r,c) << endl;
        }
    }

    Mat fused_map = Mat::zeros(size, CV_32F);
    Mat fused_conf = Mat::zeros(size, CV_32F);

    //stability_fusion(depth_maps, confidence_maps);
    for (int i=fusion_offset; i<depth_map_count-fusion_offset; ++i) {
        printf("Running confidence-based fusion for depth map %d/%d...\n",(i+1)-fusion_offset,depth_map_count-(2*fusion_offset));
        vector<Mat> offset_K(K.begin()+offset, K.begin()+end_offset);
        vector<Mat> offset_R(R.begin()+offset, R.begin()+end_offset);
        vector<Mat> offset_t(t.begin()+offset, t.begin()+end_offset);
        vector<Mat> offset_bounds(bounds.begin()+offset, bounds.begin()+end_offset);

        confidence_fusion(depth_maps, fused_map, conf_maps, fused_conf, offset_K, offset_R, offset_t, offset_bounds, i, scale, data_path);


        write_map(fused_map, data_path + "fused_maps/depth_fused_" + to_string(i) + ".csv");
        //write_map(fused_conf, "conf_fused_" + to_string(i) + ".csv");

        display_map(fused_map, "disp_depth_fused_" + to_string(i) + ".png");
        //display_map(fused_conf, "disp_conf_fused_" + to_string(i) + ".png", scale);
    }

    return EXIT_SUCCESS;
}
