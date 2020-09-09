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

// down-sample images
void down_sample(vector<Mat> *images, vector<Mat> *intrinsics, const int scale) {
    if (scale <= 0) {
        return;
    }
    Size size = (*images)[0].size();

    vector<Mat>::iterator img(images->begin());
    vector<Mat>::iterator k(intrinsics->begin());

    for (; img != images->end(); ++img,++k) {
        for (int i = 0; i < scale; ++i) {
            Mat temp_img = Mat::zeros(size, CV_32F);

            pyrDown(*img,temp_img);
            *img = temp_img;
        }

        k->at<float>(0,0) = k->at<float>(0,0)/(scale*2);
        k->at<float>(1,1) = k->at<float>(1,1)/(scale*2);
        k->at<float>(0,2) = k->at<float>(0,2)/(scale*2);
        k->at<float>(1,2) = k->at<float>(1,2)/(scale*2);
    }
}

// up-sample images (used for writing images)
Mat up_sample(const Mat *image, const int scale) {
    Size size = image->size();
    Mat enlarged_img = *image;

    for (int i = 0; i < scale; ++i) {
        Mat temp_img = Mat::zeros(size, CV_32F);
        pyrUp(enlarged_img,temp_img);
        enlarged_img = temp_img;
    }

    return enlarged_img;
}

// Image writing utility (scales to [0,255])
void write_map(const Mat map, string filename, const int scale) {
    Mat scaled_map = up_sample(&map,scale);
    double max;
    double min;
    Point min_loc;
    Point max_loc;
    minMaxLoc(scaled_map, &min, &max, &min_loc, &max_loc);
    scaled_map = scaled_map-min;
    scaled_map = (scaled_map)*(255/(max));
    imwrite(filename, scaled_map);
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
