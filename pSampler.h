//
// Created by rlav440 on 2/13/23.
//

#include <main.h>

#ifndef ACMMP_PSAMPLER_H
#define ACMMP_PSAMPLER_H
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
#endif //ACMMP_PSAMPLER_H

