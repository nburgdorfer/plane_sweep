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

using namespace cv;
using namespace std;

bool comp(char *a, char *b) {
    int res = strcmp(a,b);
    bool ret_val;

    if (res < 0) {
        ret_val = true;
    } else {
        ret_val = false;
    }

    return ret_val;
}

void load_data(vector<Mat> *images, vector<Mat> *intrinsics, vector<Mat> *rotations, vector<Mat> *translations, vector<Mat> *P, vector<Mat> *bounds, char *data_path) {
    DIR *dir;
    struct dirent *ent;

    char img_path[256];
    char camera_path[256];
    char p_path[256];
    char bounds_path[256];

    vector<char*> img_files;
    vector<char*> camera_files;
    vector<char*> p_files;
    vector<char*> bounds_files;

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


    // load intrinsics
    strcpy(camera_path,data_path);
    strcat(camera_path,"cameras/");

    FILE *fp;
    char *line=NULL;
    size_t n = 128;
    ssize_t bytes_read;

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
    char *ptr = NULL;

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

Mat plane_sweep(vector<float> &cost_volume, vector<Mat> images, int index, vector<Mat> K, vector<Mat> R, vector<Mat> t, vector<Mat> P, vector<Mat> bounds, int depth_count, int window_size) {
    Size shape = images[index].size();
    float cost;

    // compute bounds
    Mat C = t[index];

    Mat n0 = Mat::zeros(3,1,CV_32F);
    n0.at<float>(2,0) = 1;
    Mat n1 = R[index]*n0;

    Mat v_min = (bounds[index].row(1)).t() - t[index];
    Mat v_max = (bounds[index].row(0)).t() - t[index];

    float min_dist = v_min.dot(n1);
    float max_dist = v_max.dot(n1);

    float interval = (max_dist-min_dist)/depth_count;
    const int sizes[3] = {shape.height, shape.width, depth_count};

    Mat depth_map = Mat::zeros(shape.height,shape.width,CV_32F);
    Mat depth_values = Mat::zeros(shape.height,shape.width,CV_32F);
    //vector<float> cost_volume(shape.width*shape.height*depth_count);

    fill(cost_volume.begin(), cost_volume.end(), -1);
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
            mtest.push_back((-R[i].t() * t[i]).t());
            mtest = mtest.t();
            Mat ptest = K[i]*mtest;

            cout << "Mtest:\n" << mtest << endl;
            cout << "Test P:\n" <<  ptest << endl;
            cout << "True P:\n" <<  P[i] << endl<<endl;
            // TEST
            */
            
            if (index == i) {
                continue;
            }

            // compute relative extrinsics
            Mat M1;
            M1.push_back(R[index]);
            M1.push_back((-R[index].t()*t[index]).t());
            M1 = M1.t();
            Mat temp1;
            temp1.push_back(Mat::zeros(3,1,CV_32F));
            temp1.push_back(Mat::ones(1,1,CV_32F));
            M1.push_back(temp1.t());

            Mat M2;
            M2.push_back(R[i]);
            M2.push_back((-R[i].t()*t[i]).t());
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

    char loading_bar[23] = "[--------------------]";
    int lb_ind = 0;
    float perc = 0.0;
    int num_matches;

    for (int y=offset; y<shape.height-offset; ++y) {
        perc = (y/(float)(shape.height-offset)) * 100;
        lb_ind = static_cast<int>(floor(perc / 5)) + 1;
        loading_bar[lb_ind] = '#';
        cerr << "\r\tBuilding depth map: " << loading_bar << std::flush;

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
    int min_cost;
    int best_depth;
    long ind;

    for (int y=offset; y<shape.height-offset; ++y) {
        for (int x=offset; x<shape.width-offset; ++x) {
            min_cost = INT_MAX;
            best_depth = 0;

            for (float d=0,z_curr=min_dist; d<depth_count; ++d,z_curr+=interval) {
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

    double max;
    double min;
    minMaxIdx(depth_map, &min, &max);
    depth_map = (depth_map)*(255/(max*1.5));

    return depth_map;
}

void stability_fusion(){
    cout << "Running stability-based fusion..." << endl;
}


void confidence_fusion(){
    cout << "Running confidence-based fusion..." << endl;
}


int main(int argc, char **argv) {
    
    if (argc != 2) {
        fprintf(stderr, "Error: usage %s <path-to-images>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    char *data_path = argv[1];
    int depth_count = 100;
    int window_size = 11;
    
    vector<Mat> images;
    vector<Mat> depth_maps;
    vector<Mat> intrinsics;
    vector<Mat> rotations;
    vector<Mat> translations;
    vector<Mat> P;
    vector<Mat> bounds;
    vector<vector<float>> confidence_maps;
    
    // load images, Ks, Rs, ts, ps, bounds
    printf("Loading data...\n");
    load_data(&images, &intrinsics, &rotations, &translations, &P, &bounds, data_path);

    int img_count = images.size();

    // build depth map
    for (int i=0; i<img_count; ++i) {
        printf("Computing depth map for image %d/%d...\n",i+1,img_count);

        Size shape = images[i].size();
        vector<float> cost_volume(shape.width*shape.height*depth_count);

        Mat map = plane_sweep(cost_volume, images, i, intrinsics, rotations, translations, P, bounds, depth_count, window_size);

        confidence_maps.push_back(cost_volume);
        depth_maps.push_back(map);

        imwrite("depth_" + to_string(i) + ".png",map);
    }

    stability_fusion();
    confidence_fusion();



    return EXIT_SUCCESS;
}
