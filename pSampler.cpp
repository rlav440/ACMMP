//
// Created by rlav440 on 2/13/23.
//

#include "pSampler.h"
#include <memory>


pSampler::pSampler(std::string dense_loc) {

    if(using_prior){
        // load the first image
        // get the rows an cols
        int rows, cols;
        // initialise the memory of the camera array
        return;
    }

    probability_volume = nullptr;
}

pSampler::~pSampler() {

}



float4 depth_normal_to_plane(float depth, float3 normal){
    float4 temp = make_float4(0,0,0,0);
    return temp;
    // TODO replicate the logic from random init here.
}

std::unique_ptr<float4> pSampler::GetPriorPlaneEstimate(int camNum) {
    using cv::Mat;
    Mat depth = cv::imread(input_depths_list[camNum]);
    Mat normals = cv::imread(input_normals_list[camNum]);
    int rows = depth.rows;
    int cols = depth.cols;

    auto plane_hypothesis = new float4[rows * cols];
    int k = 0;
    for (int i=0; i < rows; i++){
        for (int j = 0; j < cols; ++j){
           plane_hypothesis[k] = depth_normal_to_plane(
                   depth.at<float>(i,j),
                   normals.at<float3>(i,j)
                   );
        }
    }
    return std::unique_ptr<float4>(plane_hypothesis);
}
