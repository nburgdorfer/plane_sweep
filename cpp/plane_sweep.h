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

// Data loading functions
void load_images(vector<Mat> *images, char *data_path);
void load_camera_params(vector<Mat> *intrinsics, vector<Mat> *rotations, vector<Mat> *translations, char *data_path);
void load_p_matrices(vector<Mat> *P, char *data_path);
void load_dtu_bounds(vector<Mat> *bounds, char *data_path);
void load_strecha_bounds(vector<Mat> *bounds, char *data_path);
void load_data(vector<Mat> *images, vector<Mat> *intrinsics, vector<Mat> *rotations, vector<Mat> *translations, vector<Mat> *P, vector<Mat> *bounds, char *data_path, bool r_applied);

// Plane Sweep algorithm entry function
Mat plane_sweep(vector<float> &cost_volume, const vector<Mat> &images, int index, const vector<Mat> &K, const vector<Mat> &R, const vector<Mat> &t, const vector<Mat> &P, const vector<Mat> &bounds, int depth_count, int window_size, bool r_applied);

// Confidence map construiction fucntion
void build_conf_map(const Mat &depth_map, Mat &conf_map, const vector<float> &cost_volume, Size shape, int depth_count, float sigma);

// Fusion functions
void stability_fusion(const vector<Mat> &depth_maps, const vector<Mat> &conf_maps);
void confidence_fusion(const vector<Mat> &depth_maps, const vector<Mat> &conf_maps);
