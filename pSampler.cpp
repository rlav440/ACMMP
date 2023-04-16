//
// Created by rlav440 on 2/13/23.
//

#include "pSampler.h"
#include <memory>

//#define DEBUG

pSampler::pSampler(const std::string& dense_folder, int ncams) {
    using cv::Mat;

    prior_folder = dense_folder + std::string("priors");
    depth_folder = prior_folder + std::string("/depths");
    normal_folder = prior_folder + std::string("/normals");

    // try to load the final image in the folders to assess if we are running this with priors.
    std::stringstream norm_path;
    std::stringstream depth_path;
    norm_path << normal_folder << "/" << std::setw(8) << std::setfill('0') << (ncams - 1) << ".png";
    depth_path << depth_folder << "/" << std::setw(8) << std::setfill('0') << (ncams - 1) << ".png";

    Mat test_depth = cv::imread(depth_path.str());
    Mat test_norm = cv::imread(norm_path.str());

    if (!test_depth.empty() && !test_norm.empty()){
        using_prior = true;
    }
    else
        std::cout << "couldn't load priors from "<< norm_path.str() << std::endl;

    probability_volume = nullptr;
}

pSampler::~pSampler() = default;

void normVec3 (float4 *vec)
{
    const float normSquared = vec->x * vec->x + vec->y * vec->y + vec->z * vec->z;
    const float inverse_sqrt = sqrtf (normSquared);
    vec->x *= inverse_sqrt;
    vec->y *= inverse_sqrt;
    vec->z *= inverse_sqrt;
}

void get3DPoint(const Camera camera, const int2 p, const float depth, float *X)
{
    X[0] = depth * (p.x - camera.K[2]) / camera.K[0];
    X[1] = depth * (p.y - camera.K[5]) / camera.K[4];
    X[2] = depth;
}

float distance_to_origin(Camera cam, int2 p, float depth, float4 normal){
    float X[3];
    get3DPoint(cam, p, depth, X);
    return -(normal.x * X[0] + normal.y * X[1] + normal.z * X[2]);
    // this defines planes by their minimum distance to the origin.
}


float4 getViewDirection(const Camera camera, const int2 p, const float depth)
{
    float X[3];
    get3DPoint(camera, p, depth, X);
    float norm = sqrt(X[0] * X[0] + X[1] * X[1] + X[2] * X[2]);

    float4 view_direction;
    view_direction.x = X[0] / norm;
    view_direction.y = X[1] / norm;
    view_direction.z =  X[2] / norm;
    view_direction.w = 0;
    return view_direction;
}

float4 depth_normal_to_plane(const float depth, float3 normal, int2 p, Camera camera){
    float4 temp = make_float4(normal.x,normal.y,normal.z,0);
    // loads the normal vector in world coordinates
    // makes sure it is positive in the camera vector

    float4 view_direction = getViewDirection(camera, p, depth);
    float dot_product = temp.x * view_direction.x + temp.y * view_direction.y + temp.z * view_direction.z;
    if (dot_product > 0.0f) {
        temp.x = -temp.x;
        temp.y = -temp.y;
        temp.z = -temp.z;
    }
    normVec3(&temp);

    temp.w = distance_to_origin(camera, p, depth, temp);
    return temp;
}

bool pSampler::confirm_using_prior() const {
    return using_prior;
}

float3 v3tof3(cv::Vec3f inp){
    return make_float3(inp[0], inp[1], inp[2]);
}

//float computeDepthfromPlaneHypothesis(const Camera camera, const float4
//plane_hypothesis, const int2 p)
//{
//    return -plane_hypothesis.w * camera.K[0] / (
//            (p.x - camera.K[2]) * plane_hypothesis.x +
//            (camera.K[0] / camera.K[4]) * (p.y - camera.K[5]) * plane_hypothesis.y +
//            camera.K[0] * plane_hypothesis.z);
//}

std::unique_ptr<float4> pSampler::GetPriorPlaneEstimate(
        const int camNum, Camera cam, const int rows, const int cols) {
    using cv::Mat;

    std::stringstream norm_path;
    std::stringstream depth_path;
    norm_path << normal_folder << "/" << std::setw(8) << std::setfill('0') << camNum << ".png";
    depth_path << depth_folder << "/" << std::setw(8) << std::setfill('0') << camNum << ".png";
    Mat depth = cv::imread(depth_path.str(), -1);
    Mat normals = cv::imread(norm_path.str(), -1);
    if (depth.empty() or normals.empty()){
        std::cout << "failed to load prior images: " << camNum << std::endl;
        throw std::runtime_error("error");
    }
//    cv::imshow("prior depth", depth);
//    cv::imshow("prior normals", normals);
//    cv::waitKey(-1);

    Mat conv_normals;
    normals.convertTo(conv_normals, CV_32FC3,
                      2.0/65536.0, -1.0
    );
    const float dist = (cam.depth_max - cam.depth_min);
    const float range = dist/65535.0f;
    Mat conv_depths;
    depth.convertTo(
            conv_depths, CV_32F,
            range,
            cam.depth_min
            //0
            );

//    int rows = depth.rows / scale;
//    int cols = depth.cols / scale;

    int scale = depth.rows/rows;

    std::cout << "Starting initialisation: " << rows << " " << cols << std::endl;
    auto plane_hypothesis = new float4[rows * cols];
    int k;

    cv::Mat_<float> depths_im = cv::Mat::zeros(rows, cols, CV_32FC1);
    cv::Mat_<cv::Vec3f> normals_im = cv::Mat::zeros(rows, cols, CV_32FC3);

    for (int i=0; i < rows; i++){
        for (int j = 0; j < cols; ++j){
            k = i * cols + j;

            float base_d = conv_depths.at<float>(i * scale,j * scale);
            float4 plane = depth_normal_to_plane(
                   base_d,
            v3tof3(conv_normals.at<cv::Vec3f>(i * scale,j * scale)),
                   make_int2(j,i),
                   cam
                   );
//            if((i%100 == 0) && (j%100 ==0)){
//                std::cout << "loaded depth value: " << base_d << std::endl;
//            }
//           plane = make_float4(0,0,0,0);
            plane_hypothesis[k] = plane;
            depths_im(i, j) = plane.w;
            normals_im(i,j) = cv::Vec3f(
                    plane.x, plane.y, plane.z);
        }
    }

#ifdef DEBUG
    Mat depths_vis;
    std::cout << "Depth max " << cam.depth_max << std::endl;
    depths_im.convertTo(depths_vis, CV_8U, 255.f/800);

    cv::imshow("normals loaded", normals_im);
    cv::imshow("depths loaded", depths_vis);
    cv::waitKey(-1);

    std::cout << "Loaded the prior image" << std::endl;
#endif

    return std::unique_ptr<float4>(plane_hypothesis);
}
