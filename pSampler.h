//
// Created by rlav440 on 2/13/23.
//

#include <main.h>

#ifndef ACMMP_PSAMPLER_H
#define ACMMP_PSAMPLER_H
class pSampler{
public:
    pSampler(std::string dense_loc);
    ~pSampler();

    // define the logic of the sampler

    std::unique_ptr<float4> GetPriorPlaneEstimate(int camNum);
    void UpdateProbabilityVolume();
    bool confirm_using_prior();

private:
    float4 *probability_volume;
    std::vector<std::string> input_depths_list;
    std::vector<std::string> input_normals_list;
    bool using_prior = false;



};
#endif //ACMMP_PSAMPLER_H

