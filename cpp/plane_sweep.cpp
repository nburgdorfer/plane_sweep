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

using namespace cv;
using namespace std;

/*
 * @brief Loads all files found under 'data_path' into the 'images' container
 *
 * @param images - The container to be populated with the loaded image objects
 * @param data_path - The relative path to the base directory for the data
 *
 */
void load_images(vector<Mat> *images, char *data_path) {
    DIR *dir;
    struct dirent *ent;
    char img_path[256];
    vector<char*> img_files;

    // load images
    strcpy(img_path,data_path);
    strcat(img_path,"images/");

    if((dir = opendir(img_path)) == NULL) {
        fprintf(stderr,"Error: Cannot open directory %s.\n",img_path);
        exit(EXIT_FAILURE);
    }

    while((ent = readdir(dir)) != NULL) {
        if ((ent->d_name[0] != '.') && (ent->d_type != DT_DIR)) {
            char *img_filename = (char*) malloc(sizeof(char) * 256);

            strcpy(img_filename,img_path);
            strcat(img_filename,ent->d_name);

            img_files.push_back(img_filename);
        }
    }

    // sort files by name
    sort(img_files.begin(), img_files.end(), comp);

    int img_count = img_files.size();

    for (int i=0; i<img_count; ++i) {
        Mat img = imread(img_files[i]);
        img.convertTo(img,CV_32F);
        images->push_back(img);
    }
}

/*
 * @brief Loads in the camera intrinsics and extrinsics
 *
 * @param intrinsics - The container to be populated with the intrinsic matrices for the images
 * @param rotations - The container to be populated with the rotation matrices for the images
 * @param translations - The container to be populated with the translation vectors for the images
 * @param data_path - The relative path to the base directory for the data
 *
 */
void load_camera_params(vector<Mat> *intrinsics, vector<Mat> *rotations, vector<Mat> *translations, char *data_path) {
    DIR *dir;
    struct dirent *ent;
    char camera_path[256];
    vector<char*> camera_files;

    FILE *fp;
    char *line=NULL;
    size_t n = 128;
    ssize_t bytes_read;
    char *ptr = NULL;

    // load intrinsics
    strcpy(camera_path,data_path);
    strcat(camera_path,"cameras/");

    if((dir = opendir(camera_path)) == NULL) {
        fprintf(stderr,"Error: Cannot open directory %s.\n",camera_path);
        exit(EXIT_FAILURE);
    }

    while((ent = readdir(dir)) != NULL) {
        if ((ent->d_name[0] != '.') && (ent->d_type != DT_DIR)) {
            char *camera_filename = (char*) malloc(sizeof(char) * 256);

            strcpy(camera_filename,camera_path);
            strcat(camera_filename,ent->d_name);

            camera_files.push_back(camera_filename);
        }
    }

    // sort files by name
    sort(camera_files.begin(), camera_files.end(), comp);

    int camera_count = camera_files.size();

    for (int i=0; i<camera_count; ++i) {
        if ((fp = fopen(camera_files[i],"r")) == NULL) {
            fprintf(stderr,"Error: could not open file %s.\n", camera_files[i]);
            exit(EXIT_FAILURE);
        }

        // load K matrix
        Mat K(3,3,CV_32F);
        for (int j=0; j<3; ++j) {
            if ((bytes_read = getline(&line, &n, fp)) == -1) {
                fprintf(stderr, "Error: could not read line from %s.\n",camera_files[i]);
            }

            ptr = strstr(line,"\n");
            strncpy(ptr,"\0",1);
            
            char *token = strtok(line," ");
            int ind = 0;
            while (token != NULL) {
                K.at<float>(j,ind) = atof(token);

                token = strtok(NULL," ");
                ind++;
            }
        }

        intrinsics->push_back(K);

        // throw away line "0 0 0"..... I don't know why it is there...
        // maybe for radial distortion, so unused in this algorithm...
        bytes_read = getline(&line, &n, fp);
        ptr = strstr(line,"\n");
        strncpy(ptr,"\0",1);

        // load R matrix
        Mat R(3,3,CV_32F);
        for (int j=0; j<3; ++j) {
            if ((bytes_read = getline(&line, &n, fp)) == -1) {
                fprintf(stderr, "Error: could not read line from %s.\n",camera_files[i]);
            }

            ptr = strstr(line,"\n");
            strncpy(ptr,"\0",1);
            
            char *token = strtok(line," ");
            int ind = 0;
            while (token != NULL) {
                R.at<float>(j,ind) = atof(token);

                token = strtok(NULL," ");
                ind++;
            }
        }

        rotations->push_back(R);

        // load t matrix
        Mat t(3,1,CV_32F);
        if ((bytes_read = getline(&line, &n, fp)) == -1) {
            fprintf(stderr, "Error: could not read line from %s.\n",camera_files[i]);
        }

        ptr = strstr(line,"\n");
        strncpy(ptr,"\0",1);
        
        char *token = strtok(line," ");
        int ind = 0;
        while (token != NULL) {
            t.at<float>(ind,0) = atof(token);

            token = strtok(NULL," ");
            ind++;
        }

        translations->push_back(t);

        fclose(fp);
    }
}

/*
 * @brief Loads in the boundary information based on the DTU dataset
 *
 * @param bounds - The container to be populated with the bounds information for the images
 * @param data_path - The relative path to the base directory for the data
 *
 */
void load_dtu_bounds(vector<Mat> *bounds, char *data_path) {
    DIR *dir;
    struct dirent *ent;
    char bounds_path[256];
    vector<char*> bounds_files;

    FILE *fp;
    char *line=NULL;
    size_t n = 128;
    ssize_t bytes_read;
    char *ptr = NULL;

    // load bounds
    strcpy(bounds_path,data_path);
    strcat(bounds_path,"bounding/");

    if((dir = opendir(bounds_path)) == NULL) {
        fprintf(stderr,"Error: Cannot open directory %s.\n",bounds_path);
        exit(EXIT_FAILURE);
    }

    while((ent = readdir(dir)) != NULL) {
        if ((ent->d_name[0] != '.') && (ent->d_type != DT_DIR)) {
            char *bounds_filename = (char*) malloc(sizeof(char) * 256);

            strcpy(bounds_filename,bounds_path);
            strcat(bounds_filename,ent->d_name);

            bounds_files.push_back(bounds_filename);
        }
    }

    // sort files by name
    sort(bounds_files.begin(), bounds_files.end(), comp);

    int bounds_count = bounds_files.size();

    for (int i=0; i<bounds_count; ++i) {
        if ((fp = fopen(bounds_files[i],"r")) == NULL) {
            fprintf(stderr,"Error: could not open file %s.\n", bounds_files[i]);
            exit(EXIT_FAILURE);
        }

        // load bounds vectors
        Mat bound(2,1,CV_32F);

        // get min distance
        if ((bytes_read = getline(&line, &n, fp)) == -1) {
            fprintf(stderr, "Error: could not read line from %s.\n",bounds_files[i]);
        }

        ptr = strstr(line,"\n");
        strncpy(ptr,"\0",1);
        bound.at<float>(0,0) = atof(line);

        // get max dist
        if ((bytes_read = getline(&line, &n, fp)) == -1) {
            fprintf(stderr, "Error: could not read line from %s.\n",bounds_files[i]);
        }

        ptr = strstr(line,"\n");
        strncpy(ptr,"\0",1);
        bound.at<float>(1,0) = atof(line);

        bounds->push_back(bound);

        fclose(fp);
    }
}


/*
 * @brief Loads in the boundary information based on the Strecha dataset
 *
 * @param bounds - The container to be populated with the bounds information for the images
 * @param data_path - The relative path to the base directory for the data
 *
 */
void load_strecha_bounds(vector<Mat> *bounds, char *data_path) {
    DIR *dir;
    struct dirent *ent;
    char bounds_path[256];
    vector<char*> bounds_files;

    FILE *fp;
    char *line=NULL;
    size_t n = 128;
    ssize_t bytes_read;
    char *ptr = NULL;

    // load bounds
    strcpy(bounds_path,data_path);
    strcat(bounds_path,"bounding/");

    if((dir = opendir(bounds_path)) == NULL) {
        fprintf(stderr,"Error: Cannot open directory %s.\n",bounds_path);
        exit(EXIT_FAILURE);
    }

    while((ent = readdir(dir)) != NULL) {
        if ((ent->d_name[0] != '.') && (ent->d_type != DT_DIR)) {
            char *bounds_filename = (char*) malloc(sizeof(char) * 256);

            strcpy(bounds_filename,bounds_path);
            strcat(bounds_filename,ent->d_name);

            bounds_files.push_back(bounds_filename);
        }
    }

    // sort files by name
    sort(bounds_files.begin(), bounds_files.end(), comp);

    int bounds_count = bounds_files.size();

    for (int i=0; i<bounds_count; ++i) {
        if ((fp = fopen(bounds_files[i],"r")) == NULL) {
            fprintf(stderr,"Error: could not open file %s.\n", bounds_files[i]);
            exit(EXIT_FAILURE);
        }

        // load bounds vectors
        Mat bound(2,3,CV_32F);
        for (int j=0; j<2; ++j) {
            if ((bytes_read = getline(&line, &n, fp)) == -1) {
                fprintf(stderr, "Error: could not read line from %s.\n",bounds_files[i]);
            }

            ptr = strstr(line,"\n");
            strncpy(ptr,"\0",1);
            
            char *token = strtok(line," ");
            int ind = 0;
            while (token != NULL) {
                bound.at<float>(j,ind) = atof(token);

                token = strtok(NULL," ");
                ind++;
            }
        }

        bounds->push_back(bound);

        fclose(fp);
    }
}


/*
 * @brief Loads in all the data necessary for running plane sweep
 *
 * @param images - The container to be populated with the images
 * @param intrinsics - The container to be populated with the intrinsic matrices for the images
 * @param rotations - The container to be populated with the rotation matrices for the images
 * @param translations - The container to be populated with the translation vectors for the images
 * @param P - The container to be populated with the projection matrices for the images
 * @param bounds - The container to be populated with the bounds information for the images
 * @param data_path - The relative path to the base directory for the data
 * @param dtu - Flag specifying whether or not the data is a DTU dataset
 *
 */
void load_data(vector<Mat> *images, vector<Mat> *intrinsics, vector<Mat> *rotations, vector<Mat> *translations, vector<Mat> *bounds, char *data_path, bool dtu) {
    load_images(images, data_path);
    load_camera_params(intrinsics, rotations, translations, data_path);

    if (dtu) {
        load_dtu_bounds(bounds, data_path);
    } else {
        load_strecha_bounds(bounds, data_path);
    }
}

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
    Mat depth_map = Mat::zeros(shape,CV_32F);
    Mat depth_values = Mat::zeros(shape,CV_32F);
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
//#pragma omp parallel num_threads(12) shared(homogs, dtu) private(img_homogs,M1,M2,temp1,temp2,M,H,z_curr)
//{
    // pre-compute homographies
    //#pragma omp for
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

    for (int y=offset; y<shape.height-offset; ++y) {
        for (int x=offset; x<shape.width-offset; ++x) {
            min_cost = numeric_limits<float>::max();
            best_depth = 0;

            for (float d=0; d<depth_count; ++d) {
                i = static_cast<long>(y*shape.width*depth_count) + static_cast<long>(x*depth_count) + static_cast<long>(d);
                if ((cost_volume[i] >= 0) && (cost_volume[i] < min_cost)) {
                    min_cost = cost_volume[i];
                    best_depth = d;
                }
            }
            
            depth_values.at<float>(y,x) = min_dist + (interval*best_depth);
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

float med_filt(const Mat &patch, int filter_width, int num_inliers) {
    float sum = 0.0;
    int inliers = 0;
    float initial_val = patch.at<float>((filter_width-1)/2,(filter_width-1)/2);

    for (int r=0; r<filter_width; ++r) {
        for (int c=0; c<filter_width; ++c) {
            if (patch.at<float>(r,c) >= 0) {
                sum += patch.at<float>(r,c);
                ++inliers;
            }
        }
    }

    if (inliers < num_inliers) {
        sum = initial_val;
    } else {
        sum /= inliers;
    }

    return sum;
}

void write_ply(const Mat &depth_map, const Mat &K, const Mat &P, const string filename, vector<int> color) {
    Size size = depth_map.size();

    int rows = size.height;
    int cols = size.width;

    int num_vertex = rows*cols;

    ofstream ply_file;
    ply_file.open(filename);
    ply_file << "ply\n";
    ply_file << "format ascii 1.0\n";
    ply_file << "element vertex " << num_vertex << "\n";
    ply_file << "property float x\n";
    ply_file << "property float y\n";
    ply_file << "property float z\n";
    ply_file << "property uchar red\n";
    ply_file << "property uchar green\n";
    ply_file << "property uchar blue\n";
    ply_file << "element face 0\n";
    ply_file << "end_header\n";
    

    for (int r=0; r<rows; ++r) {
        for (int c=0; c<cols; ++c) {
            float depth = depth_map.at<float>(r,c);
            if (depth == 0) {
                ply_file << "0 0 0 0 0 0\n";
                continue;
            }

            // compute corresponding (x,y) locations
            Mat x_1(3,1,CV_32F);
            x_1.at<float>(0,0) = c;
            x_1.at<float>(1,0) = r;
            x_1.at<float>(2,0) = 1;

            // take pseudo-inverse of extrinsics matrics for target image
            Mat P_inv;
            invert(P,P_inv,DECOMP_SVD);

            // find 3D world coord of back projection
            Mat X_world = P_inv * K.inv() * (depth * x_1);
            X_world.at<float>(0,0) = X_world.at<float>(0,0) / X_world.at<float>(0,3);
            X_world.at<float>(0,1) = X_world.at<float>(0,1) / X_world.at<float>(0,3);
            X_world.at<float>(0,2) = X_world.at<float>(0,2) / X_world.at<float>(0,3);
            X_world.at<float>(0,3) = X_world.at<float>(0,3) / X_world.at<float>(0,3);

            ply_file << X_world.at<float>(0,0) << " " << X_world.at<float>(0,1) << " " << X_world.at<float>(0,2) << " " << color[0] << " " << color[1] << " " << color[2] << "\n";
        }
    }

    ply_file.close();

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

    vector<Mat> P;

    cout << "\tPre-Computing Intrinsics/Extrinsics..." << endl;
    // pre-compute intrinsics/extrinsics
    for (int i=0; i<depth_map_count; ++i) {
        Mat p;
        // check to see if its a dtu dataset
        if (dtu) {
            p.push_back(R[i].t());
            p.push_back(t[i].t());
        } else {
            p.push_back(R[i]);
            p.push_back((-R[i].t()*t[i]).t());
        }
        p = p.t();

        P.push_back(p);
    }

    cout << "\tRendering depth maps into reference view..." << endl;
    vector<Mat> d_refs;
    vector<Mat> c_refs;
    const int rows = shape.height;
    const int cols = shape.width;

    for (int d=0; d < depth_map_count; ++d) {
        if (d==index) {
            d_refs.push_back(depth_maps[index]);
            c_refs.push_back(conf_maps[index]);

            // write ply file
            vector<int> green = {0, 255, 0};
            write_ply(depth_maps[index], K[index], P[index], "test_ref_" + to_string(index) + ".ply", green);

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
                Mat x_1(3,1,CV_32F);
                x_1.at<float>(0,0) = c;
                x_1.at<float>(1,0) = r;
                x_1.at<float>(2,0) = 1;

                // take pseudo-inverse of extrinsics matrics for target image
                Mat P_inv;
                invert(P[d],P_inv,DECOMP_SVD);

                // find 3D world coord of back projection
                Mat X_world = P_inv * K[d].inv() * (depth * x_1);
                X_world.at<float>(0,0) = X_world.at<float>(0,0) / X_world.at<float>(0,3);
                X_world.at<float>(0,1) = X_world.at<float>(0,1) / X_world.at<float>(0,3);
                X_world.at<float>(0,2) = X_world.at<float>(0,2) / X_world.at<float>(0,3);
                X_world.at<float>(0,3) = X_world.at<float>(0,3) / X_world.at<float>(0,3);

                // find pixel location in reference image
                Mat x_2 = K[index] * P[index] * X_world;

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

        // write ply file
        vector<int> red = {255, 0, 0};
        write_ply(d_ref, K[d], P[d], "test_" + to_string(d) + ".ply", red);

        d_refs.push_back(d_ref);
        c_refs.push_back(c_ref);

        write_map(d_ref, "d_ref_"+to_string(d)+".png", scale);
        write_map(c_ref, "c_ref_"+to_string(d)+".png", scale);
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

            d_map = d_refs.begin();
            c_map = c_refs.begin();

            for (; d_map != d_refs.end(); ++d_map,++c_map) {
                if (c_map->at<float>(r,c) > C) {
                    f = d_map->at<float>(r,c);
                    C = c_map->at<float>(r,c);
                }
            }

            initial_f = f;

            d_map = d_refs.begin();
            c_map = c_refs.begin();

            // for each depth map:
            for (; d_map != d_refs.end(); ++d_map,++c_map) {

                // if depth is close to initial depth:
                if (abs(d_map->at<float>(r,c) - initial_f) < eps) {
                    if((C + c_map->at<float>(r,c)) == 0) {
                        f = 0;
                    } else {
                        f = (f*C + d_map->at<float>(r,c)*c_map->at<float>(r,c)) / (C + c_map->at<float>(r,c));
                    }
                    C += c_map->at<float>(r,c);
                } 
                // if depth is too close (occlusion):
                else if(d_map->at<float>(r,c) < initial_f) {
                    C -= c_map->at<float>(r,c);
                }
                // if depth is too large (free space violation):
                else if(d_map->at<float>(r,c) > initial_f) {
                    C -= c_map->at<float>(r,c);
                    // C -= C_i(P(X))
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

    int w = 9;
    int w_offset = (w-1)/2;
    int w_inliers = (w*w)/2;

    int w_s = 5;
    int w_s_offset = (w_s-1)/2;
    int w_s_inliers = (w_s*w_s)/2;

    // Fill in holes (-1 values) in depth map
    for (int r=w_offset; r<rows-w_offset; ++r) {
        for (int c=w_offset; c<cols-w_offset; ++c) {
            if (fused_map.at<float>(r,c) < 0.0){
                fused_map.at<float>(r,c) = med_filt(fused_map(Rect(c-w_offset,r-w_offset,w,w)), w, w_inliers);
            }
        }
    }

    // Smooth out inliers
    for (int r=w_s_offset; r<rows-w_s_offset; ++r) {
        for (int c=w_s_offset; c<cols-w_s_offset; ++c) {
            if (fused_map.at<float>(r,c) != -1){
                fused_map.at<float>(r,c) = med_filt(fused_map(Rect(c-w_s_offset,r-w_s_offset,w_s,w_s)), w_s, w_s_inliers);
            }
        }
    }

    write_map(fused_map, "depth_fused_" + to_string(index) + ".png", scale);
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

    // build depth map
    for (int i=offset; i<img_count-offset; ++i) {
        printf("Computing depth map for image %d/%d...\n",i,img_count-2);

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

    //stability_fusion(depth_maps, confidence_maps);
    for (int i=0; i<depth_map_count; ++i) {
        printf("Running confidence-based fusion for depth map %d/%d...\n",i+1,depth_map_count);
        vector<Mat> offset_K(K.begin()+offset, K.end()-offset);
        vector<Mat> offset_R(R.begin()+offset, R.end()-offset);
        vector<Mat> offset_t(t.begin()+offset, t.end()-offset);
        vector<Mat> offset_bounds(bounds.begin()+offset, bounds.end()-offset);


        confidence_fusion(depth_maps, confidence_maps, offset_K, offset_R, offset_t, offset_bounds, i, depth_count, shape, dtu, window_size, scale);
    }

    return EXIT_SUCCESS;
}
