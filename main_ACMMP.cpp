//
// Created by rlav440 on 6/19/23.
//

#include "acmmp_definitions.h"
#include "boost/program_options.hpp"
#include "ACMMP.h"

int main(int argc, char** argv)
{
    namespace po = boost::program_options;
    std::string output_dir = "/ACMMP";
    std::string fusion_dir = "/ACMMP";
    std::string mask_dir = " ";
    std::string image_dir = "/images";

    float consistency_scalar = 0.3;
    int num_consistent_thresh = 1;
    int single_match_penalty = 0;

    po::options_description desc("allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("prior,p", "runs the reconstruction from a provided prior")
            ("fuse_thresh,f",
             po::value<float>(&consistency_scalar),
             "Sets the average inverse score threshold for fusion")
            ("dense_folder",
             po::value<std::string>(),
             "The input folder for reconstruction")
            ("multi_fusion",
             po::value<std::string>(&fusion_dir)->implicit_value("/ACMMP"),
             "Use information from a previous reconstruction during fusion of "
             "invididual camera reconstructions")
            ("force_fusion", "forces multi fusion, without prior")
            ("output_dir", po::value<std::string>(&output_dir)->implicit_value("/ACCMP"),
             "Output working directory name")
            ("num_consistent_thresh", po::value<int>(&num_consistent_thresh),
             "Number of points that must be consistent to be fused into the "
             "final output point cloud.")
            ("single_match_penalty", po::value<int>(&single_match_penalty),
             "An increase to the consistency threshold for matched "
             "hypotheses that only matched over a single set")
            ("mask_dir", po::value<std::string>(&mask_dir),
             "Directory of boolean masks (0, 255)")
            ("image_override", po::value<std::string>(&image_dir),
             "A new directory to pull texture information from, rather than "
             "the default images when fusing.")
            ;
    po::positional_options_description p;
    p.add("dense_folder", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
            options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help")){
        std::cout << desc << "\n";
        return 1;
    }

    bool prior = (bool) vm.count("prior");
    std::string dense_folder = vm["dense_folder"].as<std::string>();
    std::vector<Problem> problems;
    GenerateSampleList(dense_folder, problems);
    size_t num_images = problems.size();
    std::cout << "There are " << num_images << " problems needed to be processed!" << std::endl;

    int max_num_downscale = ComputeMultiScaleSettings(dense_folder, problems);
    pSampler sample_handler(dense_folder, num_images);

    if (prior && !sample_handler.confirm_using_prior()){
        std::cout << "Initialisation from a prior was requested, but no "
                     "suitable priors were found." << std::endl;
        return -1;
    }
    bool renamed_outdir = (bool)vm.count("output_dir");
    std::string out_name;
    if (prior && !renamed_outdir){
        out_name = "/ACMMP_PRIOR";
    }
    else
        out_name = output_dir;
    std::string output_folder = dense_folder + out_name;
    mkdir(output_folder.c_str(), 0777);


    int flag = 0;
    int geom_iterations = 2;
    bool geom_consistency = false;
    bool planar_prior = false;
    bool hierarchy = false;
    bool multi_geometry = false;

    while (max_num_downscale >= 0) {
        std::cout << "Scale: " << max_num_downscale << std::endl;

        for (size_t i = 0; i < num_images; ++i) {
            if (problems[i].num_downscale >= 0) {
                problems[i].cur_image_size = problems[i].max_image_size / pow
                        (2, problems[i].num_downscale);
                problems[i].num_downscale--;
            }
        }

        if (flag == 0) {
            flag = 1;
            geom_consistency = false;
            planar_prior = true;

            for (size_t i = 0; i < num_images; ++i) {
                //if seeded, run ingest, and load the data to the pointer.
                ProcessProblem(
                        sample_handler,
                        output_folder, dense_folder,
                        problems, i, geom_consistency,
                        planar_prior, hierarchy, multi_geometry, prior
                );
            }
            geom_consistency = true;
            planar_prior = false;
            for (int geom_iter = 0; geom_iter < geom_iterations; ++geom_iter) {
                if (geom_iter == 0) {
                    multi_geometry = false;
                }
                else {
                    multi_geometry = true;
                }
                for (size_t i = 0; i < num_images; ++i) {
                    ProcessProblem(
                            sample_handler,
                            output_folder, dense_folder,
                            problems, i, geom_consistency,
                            planar_prior, hierarchy, multi_geometry
                    );
                }
            }
        }
        else {
            std::cout << "Starting JBU" << std::endl;
            for (size_t i = 0; i < num_images; ++i) {
                JointBilateralUpsampling(dense_folder, output_folder, problems[i],
                                         problems[i].cur_image_size);
            }
            hierarchy = true;
            geom_consistency = false;
            planar_prior = true;
            for (size_t i = 0; i < num_images; ++i) {
                ProcessProblem(
                        sample_handler,
                        output_folder, dense_folder,
                        problems, i, geom_consistency, planar_prior, hierarchy
                );
            }
            hierarchy = false;
            geom_consistency = true;
            planar_prior = false;
            for (int geom_iter = 0; geom_iter < geom_iterations; ++geom_iter) {
                if (geom_iter == 0) {
                    multi_geometry = false;
                }
                else {
                    multi_geometry = true;
                }
                for (size_t i = 0; i < num_images; ++i) {
                    ProcessProblem(
                            sample_handler,
                            output_folder, dense_folder,
                            problems, i, geom_consistency,
                            planar_prior, hierarchy, multi_geometry);
                }
            }
        }
        max_num_downscale--;
    }
    geom_consistency = true;
    bool multi_aware = (bool)vm.count("multi_fusion");
    bool force_fusion = (bool)vm.count("force_fusion");
    std::string fusion_folder = dense_folder + fusion_dir;
    if ((prior && multi_aware) | force_fusion){
        RunPriorAwareFusion(
                dense_folder, output_folder, fusion_folder, problems,
                geom_consistency,
                consistency_scalar, num_consistent_thresh, single_match_penalty,
                mask_dir

        );
    }
    else{
        RunFusion(
                dense_folder, output_folder, problems, geom_consistency,
                consistency_scalar, num_consistent_thresh,
                image_dir, mask_dir
        );
    }
    return 0;
}
