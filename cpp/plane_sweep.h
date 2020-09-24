#include <vector>

using namespace std;
using namespace cv;


// Plane Sweep algorithm entry function
Mat plane_sweep(vector<float> &cost_volume, const vector<Mat> &images, int index, const vector<Mat> &K, const vector<Mat> &R, const vector<Mat> &t, const vector<Mat> &P, const vector<Mat> &bounds, int depth_count, int window_size, bool r_applied);

// Confidence map construiction fucntion
void build_conf_map(const Mat &depth_map, Mat &conf_map, const vector<float> &cost_volume, Size shape, int depth_count, float sigma);

// Fusion functions
void stability_fusion(const vector<Mat> &depth_maps, const vector<Mat> &conf_maps);
void confidence_fusion(const vector<Mat> &depth_maps, const vector<Mat> &conf_maps);
