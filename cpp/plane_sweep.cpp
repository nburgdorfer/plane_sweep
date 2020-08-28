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
 * @brief Loads in the projection matrices
 *
 * @param P - The container to be populated with the projection matrices for the images
 * @param data_path - The relative path to the base directory for the data
 *
 */
void load_p_matrices(vector<Mat> *P, char *data_path) {
    DIR *dir;
    struct dirent *ent;
    char p_path[256];
    vector<char*> p_files;

    FILE *fp;
    char *line=NULL;
    size_t n = 128;
    ssize_t bytes_read;
    char *ptr = NULL;

    // load P
    strcpy(p_path,data_path);
    strcat(p_path,"p/");

    if((dir = opendir(p_path)) == NULL) {
        fprintf(stderr,"Error: Cannot open directory %s.\n",p_path);
        exit(EXIT_FAILURE);
    }

    while((ent = readdir(dir)) != NULL) {
        if ((ent->d_name[0] != '.') && (ent->d_type != DT_DIR)) {
            char *p_filename = (char*) malloc(sizeof(char) * 256);

            strcpy(p_filename,p_path);
            strcat(p_filename,ent->d_name);

            p_files.push_back(p_filename);
        }
    }

    // sort files by name
    sort(p_files.begin(), p_files.end(), comp);

    int p_count = p_files.size();

    for (int i=0; i<p_count; ++i) {
        if ((fp = fopen(p_files[i],"r")) == NULL) {
            fprintf(stderr,"Error: could not open file %s.\n", p_files[i]);
            exit(EXIT_FAILURE);
        }

        // load p matrices
        Mat p_mat(3,4,CV_32F);
        for (int j=0; j<3; ++j) {
            if ((bytes_read = getline(&line, &n, fp)) == -1) {
                fprintf(stderr, "Error: could not read line from %s.\n",p_files[i]);
            }

            ptr = strstr(line,"\n");
            strncpy(ptr,"\0",1);
            
            char *token = strtok(line," ");
            int ind = 0;
            while (token != NULL) {
                p_mat.at<float>(j,ind) = atof(token);

                token = strtok(NULL," ");
                ind++;
            }
        }

        P->push_back(p_mat);

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
 * @brief
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
void load_data(vector<Mat> *images, vector<Mat> *intrinsics, vector<Mat> *rotations, vector<Mat> *translations, vector<Mat> *P, vector<Mat> *bounds, char *data_path, bool dtu) {
    load_images(images, data_path);
    load_camera_params(intrinsics, rotations, translations, data_path);
    load_p_matrices(P, data_path);

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
Mat plane_sweep(vector<float> &cost_volume, const vector<Mat> &images, const vector<Mat> &K, const vector<Mat> &R, const vector<Mat> &t, const vector<Mat> &P, const vector<Mat> &bounds, int index, int depth_count, int window_size, bool dtu) {
    Size shape = images[index].size();
    float cost;

    // compute bounds
    Mat C = t[index];
    float min_dist;
    float max_dist;

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
    const int sizes[3] = {shape.height, shape.width, depth_count};

    Mat depth_map = Mat::zeros(shape.height,shape.width,CV_32F);
    Mat depth_values = Mat::zeros(shape.height,shape.width,CV_32F);
    //vector<float> cost_volume(shape.width*shape.height*depth_count);

    fill(cost_volume.begin(), cost_volume.end(), numeric_limits<float>::max());
    depth_map.setTo(0);

    // compute plane norm
    Mat n = Mat::zeros(3,1,CV_32F);
    n.at<float>(2,0) = 1;

    int img_count = images.size();
    int offset = (window_size-1)/2;

    vector<vector<Mat>> homogs;

    cout << "\tPre-computing homographies..." << endl;
    // pre-compute homographies
    for (float z_curr = min_dist,d=0; d<depth_count; z_curr+=interval,++d) {
        vector<Mat> img_homogs;
        for (int i=0; i<img_count; ++i) {
    
            /*
            // TEST
            Mat mtest;
            mtest.push_back(R[i]);
            mtest.push_back((-R[i].t()*t[i]).t());
            mtest = mtest.t();
            Mat ptest = K[i]*mtest;

            cout << "Mtest:\n" << mtest << endl;
            cout << "Test P:\n" <<  ptest << endl;
            //cout << "True P:\n" <<  P[i] << endl<<endl;
            // TEST
            */
            
            if (index == i) {
                continue;
            }

            // compute relative extrinsics
            Mat M1;
            // check to see if rotation matrix was already applied to translation vector
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
            // check to see if rotation matrix was already applied to translation vector
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

            /*
            // TEST
            Mat X_center_dot(4,1,CV_32F);
            X_center_dot.at<float>(0,0) = -16.585;
            X_center_dot.at<float>(1,0) = -10.9;
            X_center_dot.at<float>(2,0) = -2.04;
            X_center_dot.at<float>(3,0) = 1;

            Mat X_spiral(4,1,CV_32F);
            X_spiral.at<float>(0,0) = -18.5579;
            X_spiral.at<float>(1,0) = -10.1996;
            X_spiral.at<float>(2,0) = -0.838706;
            X_spiral.at<float>(3,0) = 1;

            Mat X_rc(4,1,CV_32F);
            X_rc.at<float>(0,0) = -14.2707;
            X_rc.at<float>(1,0) = -11.4276;
            X_rc.at<float>(2,0) = 0.29232;
            X_rc.at<float>(3,0) = 1;

            Mat X_lc(4,1,CV_32F);
            X_lc.at<float>(0,0) = -18.7557;
            X_lc.at<float>(1,0) = -10.0207;
            X_lc.at<float>(2,0) = 0.288476;
            X_lc.at<float>(3,0) = 1;

            Mat X_flc(4,1,CV_32F);
            X_flc.at<float>(0,0) = -16.919;
            X_flc.at<float>(1,0) = -8.67291;
            X_flc.at<float>(2,0) = 0.652727;
            X_flc.at<float>(3,0) = 1;

            Mat x_i = P[i]*X_flc;
            x_i.at<float>(0,0) = x_i.at<float>(0,0)/x_i.at<float>(2,0);
            x_i.at<float>(1,0) = x_i.at<float>(1,0)/x_i.at<float>(2,0);
            x_i.at<float>(2,0) = x_i.at<float>(2,0)/x_i.at<float>(2,0);

            Mat x_ref = P[index]*X_flc;
            x_ref.at<float>(0,0) = x_ref.at<float>(0,0)/x_ref.at<float>(2,0);
            x_ref.at<float>(1,0) = x_ref.at<float>(1,0)/x_ref.at<float>(2,0);
            x_ref.at<float>(2,0) = x_ref.at<float>(2,0)/x_ref.at<float>(2,0);

            cout << "\ttrg coords:\n" << x_i << endl;
            cout << "\tref coords:\n" << x_ref << endl;


            Mat n0 = Mat::zeros(3,1,CV_32F);
            n0.at<float>(2,0) = 1;
            Mat n1 = R[index]*n0;

            Mat v = X_flc(Rect(0,0,1,3)) - t[index];

            float d = v.dot(n1);
            
            Mat H = K[i]*(R_rel + (t_rel * n.t()/d)) * K[index].inv();

            Mat x = H*x_ref;
            x.at<float>(0,0) = x.at<float>(0,0)/x.at<float>(2,0);
            x.at<float>(1,0) = x.at<float>(1,0)/x.at<float>(2,0);
            x.at<float>(2,0) = x.at<float>(2,0)/x.at<float>(2,0);


            cout << "\tgt dist:\n" << d << endl;
            cout << "\thomography est:\n" << x << endl << endl;

            // Test
            */

            img_homogs.push_back(H);
        }
        homogs.push_back(img_homogs);
    }

    int num_matches;

    cout << "\tBuilding depth map..." << endl;
    for (int y=offset; y<shape.height-offset; ++y) {
        for (int x=offset; x<shape.width-offset; ++x) {
            for (float d=0,z_curr=min_dist; d<depth_count; ++d,z_curr+=interval) {
                cost = 0;
                num_matches = 0;
                
                for (int i=0,ind=0; i<img_count; ++i,++ind) {
                    if (index == i) {
                        ind--;
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
                }

                if(num_matches > 0) {
                    long ind = (long) (y*shape.width*depth_count) + (long) (x*depth_count) + (long) d;
                    cost_volume[ind] = cost/num_matches;
                }
            }
        }
    }


    // build depth map
    float min_cost;
    int best_depth;
    long ind;

    for (int y=offset; y<shape.height-offset; ++y) {
        for (int x=offset; x<shape.width-offset; ++x) {
            min_cost = numeric_limits<float>::max();
            best_depth = 0;

            for (float d=0; d<depth_count; ++d) {
                ind = (long) (y*shape.width*depth_count) + (long) (x*depth_count) + (long) d;
                if ((cost_volume[ind] >= 0) && (cost_volume[ind] < min_cost)) {
                    min_cost = cost_volume[ind];
                    best_depth = d;
                }
            }
            
            depth_map.at<float>(y,x) = best_depth;
            depth_values.at<float>(y,x) = min_dist + (interval*best_depth);
        }
    }


    return depth_map;
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
void build_conf_map(const Mat &depth_map, Mat &conf_map, const vector<float> &cost_volume, const Size shape, const int depth_count, const float sigma) {
    cout << "\tBuilding confidence map..." << endl;
    int rows = shape.height;
    int cols = shape.width;
    long ind;
    long ind_0;
    long conf_ind;
    float conf_sum;
    float curr_depth;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            conf_sum = 0.0;

            for(int d = 0; d < depth_count; ++d) {
                curr_depth = depth_map.at<float>(r,c);
                if (curr_depth == d) {
                    continue;
                }
                
                ind = static_cast<long>(r*cols*depth_count) + static_cast<long>(c*depth_count) + static_cast<long>(d);
                ind_0 = static_cast<long>(r*cols*depth_count) + static_cast<long>(c*depth_count) + static_cast<long>(curr_depth);
                conf_sum += exp(-(pow(cost_volume[ind] - cost_volume[ind_0], 2) / pow(sigma,2)));
            }
            conf_sum = pow(conf_sum, -1);
            if (conf_sum == numeric_limits<float>::infinity()) {
                conf_sum = numeric_limits<float>::max();
            }
            conf_ind = static_cast<long>(r*cols) + static_cast<long>(c);
            conf_map.at<float>(r,c) = conf_sum;
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
void confidence_fusion(const vector<Mat> &depth_maps, const vector<Mat> &conf_maps){
    cout << "Running confidence-based fusion..." << endl;
    // TODO: Render all depth maps into reference view
    
    // TODO: Render all confidence maps into reference view
    
    // TODO: Fuse depth maps
    // for each pixel:
        // set initial depth estimate and confidence value
        // for each depth map:
            // if depth is close to initial depth:
                // f = (f*C + d_ref_i(r,c)*C_ref_i(r,c)) / C + C_ref_i(r,c)
                // C = C + C_ref_i(r,c)
            // if depth is too close (occlusion):
                // C = C - C_ref_i(r,c)
            // if depth is too large (free space violation):
                // C = C - C_i(P(X))
        // if C < 0:
            // C = -1;
            
    // TODO: Fill in holes (-1 values) in depth map:
    // for each pixel:
        // if pixel value == -1:
            // perform median filter using only inliers (non -1 valued pixels) with window size 'w x w'
    // TODO: Smooth out inliers:
        // for each pixel:
            //  if pixel value >= 0:
               // perform median filter using only inliers with small window size 'w_s x w_s'
}


int main(int argc, char **argv) {
    
    if (argc != 4) {
        fprintf(stderr, "Error: usage %s <0 if camera centers / 1 if translation vectors> <path-to-images> <sigma>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    bool dtu = atoi(argv[1]);
    char *data_path = argv[2];
    float sigma = atof(argv[3]);
    int depth_count = 10;
    int window_size = 3;
    size_t str_len = strlen(data_path);

    if (data_path[str_len-1] != '/') {
        strcat(data_path, "/");
    }
    
    vector<Mat> images;
    vector<Mat> depth_maps;
    vector<Mat> intrinsics;
    vector<Mat> rotations;
    vector<Mat> translations;
    vector<Mat> P;
    vector<Mat> bounds;
    vector<Mat> confidence_maps;
    
    // load images, K's, R's, t's, P's, bounds
    printf("Loading data...\n");
    load_data(&images, &intrinsics, &rotations, &translations, &P, &bounds, data_path, dtu);

    int img_count = images.size();

    // build depth map
    for (int i=1; i<img_count-1; ++i) {
        printf("Computing depth map for image %d/%d...\n",i+1,img_count);

        Size shape = images[i].size();
        vector<float> cost_volume(shape.width*shape.height*depth_count);

        Mat map = plane_sweep(cost_volume, images, intrinsics, rotations, translations, P, bounds, i, depth_count, window_size, dtu);

        Mat conf_map = Mat::zeros(shape, CV_32F);
        build_conf_map(map, conf_map, cost_volume, shape, depth_count, sigma);
        confidence_maps.push_back(conf_map);
        depth_maps.push_back(map);

        // shift range of depth values and write image
        double max;
        double min;
        Point min_loc;
        Point max_loc;
        minMaxLoc(map, &min, &max, &min_loc, &max_loc);
        Mat depth_map = (map)*(255/(max*1.5));
        imwrite("depth_" + to_string(i) + ".png", depth_map);

        // shift range of confidence values and write image
        max;
        min;
        min_loc;
        max_loc;
        cv::minMaxLoc(conf_map, &min, &max, &min_loc, &max_loc);
        Mat c_map = (conf_map)*(255/(max*1.5));
        imwrite("conf_" + to_string(i) + ".png", c_map);
    }

    //stability_fusion(depth_maps, confidence_maps);
    confidence_fusion(depth_maps, confidence_maps);

    return EXIT_SUCCESS;
}
