//
// Created by rlav440 on 6/18/23.
//
#include "acmmp_definitions.h"
#include "boost/program_options.hpp"

int main(int argc, char** argv)
{
    namespace po = boost::program_options;
    std::string output_dir = "/ACMMP";
    std::string fusion_dir = "/ACMMP";
    std::string mask_dir = " ";
    std::string image_override = "/images";

    float consistency_scalar = 0.3;
    int num_consistent_thresh = 1;
    int single_match_penalty = 0;

    po::options_description desc("allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("fuse_thresh,f",
             po::value<float>(&consistency_scalar),
             "Sets the average inverse score threshold for fusion")
            ("dense_folder",
             po::value<std::string>(),
             "The default location for ACMMP outputs to fuse")
            ("multi_fusion",
             po::value<std::string>(&fusion_dir)->implicit_value("/ACMMP"),
             "Use information from additional reconstructions")
            ("force_fusion", "forces multi fusion, without prior")
            ("num_consistent_thresh", po::value<int>(&num_consistent_thresh),
             "Number of points that must be consistent to be fused into the "
             "final output point cloud.")
            ("single_match_penalty", po::value<int>(&single_match_penalty),
             "An increase to the consistency threshold for matched "
             "hypotheses that only matched over a single set")
            ("mask_dir", po::value<std::string>(&mask_dir),
                    "Directory of boolean masks (0, 255) for masking fusion")
            ("image_override", po::value<std::string>(&image_override),
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

    bool renamed_outdir = (bool)vm.count("output_dir");
    std::string out_name;
    if (prior && !renamed_outdir){
        out_name = "/ACMMP_PRIOR";
    }
    else
        out_name = output_dir;
    std::string output_folder = dense_folder + out_name;
    mkdir(output_folder.c_str(), 0777);

    bool geom_consistency = false;
    bool multi_aware = (bool)vm.count("multi_fusion");
    bool force_fusion = (bool)vm.count("force_fusion");

    std::string fusion_folder = dense_folder + fusion_dir;
    if ((prior && multi_aware) | force_fusion){
        RunPriorAwareFusion(
                dense_folder, output_folder, fusion_folder, problems,
                geom_consistency,
                consistency_scalar, num_consistent_thresh, single_match_penalty,
                image_override
        );
    }
    else{
        RunFusion(
                dense_folder, output_folder, problems, geom_consistency,
                consistency_scalar, num_consistent_thresh,
                image_override,  mask_dir
        );
    }
    return 0;
}
