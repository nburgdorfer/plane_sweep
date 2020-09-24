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

void load_images(vector<Mat> *images, char *data_path);
void load_camera_params(vector<Mat> *K, vector<Mat> *R, vector<Mat> *t, char *data_path);
void load_dtu_bounds(vector<Mat> *bounds, char *data_path);
void load_strecha_bounds(vector<Mat> *bounds, char *data_path);
void load_data(vector<Mat> *images, vector<Mat> *K, vector<Mat> *R, vector<Mat> *t, vector<Mat> *bounds, char *data_path, bool dtu);
float med_filt(const Mat &patch, int filter_width, int num_inliers);
float mean_filt(const Mat &patch, int filter_width, int num_inliers);
void write_ply(const Mat &depth_map, const Mat &K, const Mat &P, const string filename, const vector<int> color);
void down_sample(vector<Mat> *images, vector<Mat> *intrinsics, const int scale);
Mat up_sample(const Mat *image, const int scale);
void write_map(const Mat map, string filename, const int scale);
