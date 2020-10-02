#ifndef _UTIL_H_
#define _UTIL_H_

#include <vector>

using namespace std;
using namespace cv;

// String comparison function
inline bool comp(char *a, char *b) {
    int res = strcmp(a,b);
    bool ret_val;

    if (res < 0) {
        ret_val = true;
    } else {
        ret_val = false;
    }

    return ret_val;
}

void load_conf_maps(vector<Mat> *conf_maps, char *data_path);
void load_depth_maps(vector<Mat> *depth_maps, char *data_path);
void load_images(vector<Mat> *images, char *data_path);
void load_camera_params(vector<Mat> *K, vector<Mat> *R, vector<Mat> *t, char *data_path);
void load_bounds(vector<Mat> *bounds, char *data_path);
float med_filt(const Mat &patch, int filter_width, int num_inliers);
float mean_filt(const Mat &patch, int filter_width, int num_inliers);
void write_ply(const Mat &depth_map, const Mat &K, const Mat &P, const string filename, const vector<int> color);
void down_sample(vector<Mat> *images, vector<Mat> *intrinsics, const int scale);
void down_sample_k(vector<Mat> *intrinsics, const int scale);
Mat up_sample(const Mat *image, const int scale);
void display_map(const Mat map, string filename, const int scale);
void write_map(const Mat map, string filename);

#endif
