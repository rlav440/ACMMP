#ifndef _MAIN_H_
#define _MAIN_H_

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <memory>
#include "iomanip"

#include <sys/stat.h> // mkdir
#include <sys/types.h> // mkdir

#define MAX_IMAGES 256
#define JBU_NUM 2

struct Camera {
    float K[9];
    float R[9];
    float t[3];
    int height;
    int width;
    float depth_min;
    float depth_max;
};

struct Problem {
    int ref_image_id;
    std::vector<int> src_image_ids;
    int max_image_size = 3200;
    int num_downscale = 0;
    int cur_image_size = 3200;
};

struct Triangle {
    cv::Point pt1, pt2, pt3;
    Triangle (const cv::Point _pt1, const cv::Point _pt2, const cv::Point _pt3) : pt1(_pt1) , pt2(_pt2), pt3(_pt3) {}
};

struct PointList {
    float3 coord;
    float3 normal;
    float3 color;
};

class pSampler{
public:
    pSampler(const std::string& dense_loc, const int nCam);
    ~pSampler();

    // define the logic of the sampler

    std::unique_ptr<float4> GetPriorPlaneEstimate(int camNum, Camera cam,
                                                  int rows, int cols);
    // void UpdateProbabilityVolume();
    bool confirm_using_prior() const;

private:
    float4 *probability_volume;
    bool using_prior = false;
    std::string prior_folder;
    std::string depth_folder;
    std::string normal_folder;

};


void GenerateSampleList(const std::string &dense_folder, std::vector<Problem>
        &problems);

int ComputeMultiScaleSettings(const std::string &dense_folder, std::vector<Problem> &problems);

void ProcessProblem(
        pSampler &pSample, const std::string output_folder,
        const std::string &dense_folder, const std::vector<Problem> &problems, const int idx,
        bool geom_consistency, bool planar_prior, bool hierarchy, bool multi_geometry=false, bool seeded=false
        );

void JointBilateralUpsampling(
        const std::string &dense_folder,
        const std::string &output_folder,
        const Problem &problem, int acmmp_size
        );

void RunFusion(std::string &dense_folder, std::string &outfolder,
               const std::vector<Problem> &problems, bool geom_consistency,
               float consistency_scalar, int con_num_thresh);

void RunPriorAwareFusion(
        std::string &dense_folder, std::string &outfolder,
        std::string &fusion_folder,
        const std::vector<Problem> &problems, bool geom_consistency, float
        consistency_scalar, int num_consistent_thresh,
        int single_match_penalty = 1);
#endif // _MAIN_H_
