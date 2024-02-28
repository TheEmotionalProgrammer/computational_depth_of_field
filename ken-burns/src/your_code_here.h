#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <tuple>
#include <vector>

#include <stdexcept>
#include <opencv2/opencv.hpp>


#include "helpers.h"
#include <omp.h>



//CREATE VIDEO FROM A LIST OF FRAMES
void createVideo(std::vector<cv::Mat> frames, std::string filename, int fps){
    cv::Size frameSize(frames[0].cols, frames[0].rows);
    //create an mp4 video
    cv::VideoWriter video(filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, frameSize);
    for (int i = 0; i < frames.size(); i++){
        video.write(frames[i]);
    }
    video.release();
}

template <typename T>
inline T sampleBilinear(const Image<T>& image, const glm::vec2& pos_px)
{
    // Calculate the coordinates of the pixel centers around the sample point.
    float x = pos_px.x - 0.5f; // Account for pixel center offset
    float y = pos_px.y - 0.5f;
    
    // Determine the coordinates of the top-left pixel.
    int x0 = glm::clamp(static_cast<int>(glm::floor(x)), 0, image.width - 1);
    int y0 = glm::clamp(static_cast<int>(glm::floor(y)), 0, image.height - 1);
    
     // Determine the coordinates of the bottom-right pixel.
    int x1 = glm::clamp(x0 + 1, 0, image.width - 1);
    int y1 = glm::clamp(y0 + 1, 0, image.height - 1);
    
    auto A = image.data[y0 * image.width + x0];
    auto B = image.data[y0 * image.width + x1];
    auto C = image.data[y1 * image.width + x0];
    auto D = image.data[y1 * image.width + x1];

    float alpha = x - x0;
    float beta = y1 - y;

    auto fx0 = (1 - alpha) * A + alpha * B;
    auto fx1 = (1 - alpha) * C + alpha * D;
    T result = (1 - beta) * fx1 + beta * fx0;

    return result;
}


/*
  Core functions.
*/

/// <summary>
/// Applies the bilateral filter on the given disparity image.
/// Ignored pixels that are marked as invalid.
/// </summary>
/// <param name="disparity">The image to be filtered.</param>
/// <param name="guide">The image guide used for calculating the tonal distances between pixel values.</param>
/// <param name="size">The kernel size, which is always odd.</param>
/// <param name="guide_sigma">Sigma value of the gaussian guide kernel.</param>
/// <returns>ImageFloat, the filtered disparity.</returns>
ImageFloat jointBilateralFilter(const ImageFloat& disparity, const ImageRGB& guide, const int size, const float guide_sigma)
{   
    

    // The filter size is always odd.
    assert(size % 2 == 1);

    // We assume both images have matching dimensions.
    assert(disparity.width == guide.width && disparity.height == guide.height);

    // Rule of thumb for gaussian's std dev. 
    const float sigma = (size - 1) / 2 / 3.2f;

    // Empty output image.
    auto result = ImageFloat(disparity.width, disparity.height);
    std::fill(result.data.begin(), result.data.end(), 0.0f);
    #pragma omp parallel for 
    for (int y = 0; y< disparity.height; y++){
        for (int x = 0; x<disparity.width; x++){

            float sum_weight = 0.0f;
            for (int i = 0; i < size; i++){
                for (int j = 0; j < size; j++){

                    int x0 = x - size / 2 + i;
                    int y0 = y - size / 2 + j;

                    if (x0 < 0 || x0 >= disparity.width || y0 < 0 || y0 >= disparity.height || disparity.data[y0 * disparity.width + x0] == INVALID_VALUE){
                        continue;
                    }

                    float dist = glm::distance(glm::vec2(x, y), glm::vec2(x0, y0));
                    float diff_value = glm::distance(guide.data[y * guide.width + x], guide.data[y0 * guide.width + x0]);

                    float w_i = gauss(dist, sigma) * gauss(diff_value, guide_sigma);
                    sum_weight += w_i;
                    result.data[y * result.width + x] += w_i * disparity.data[y0 * disparity.width + x0];
                    
                }
            }
            if (sum_weight == 0.0f){
                result.data[y * result.width + x] = INVALID_VALUE;
            }
            else{
                //Compute weighted mean of all (unskipped) neighboring pixel disparities and assign it to the output.
                result.data[y * result.width + x] /= sum_weight;
            }
            


        }



    }
    return result;
    
}

/// <summary>
/// In-place normalizes and an ImageFloat image to be between 0 and 1.
/// All values marked as invalid will stay marked as invalid.
/// </summary>
/// <param name="scalar_image"></param>
/// <returns></returns>
void normalizeValidValues(ImageFloat& scalar_image)
{

    float min_value = std::numeric_limits<float>::max();
    float max_value = std::numeric_limits<float>::min();

    #pragma omp parallel for
    for (int i = 0; i < scalar_image.data.size(); i++){
        if (scalar_image.data[i] != INVALID_VALUE){
            if (scalar_image.data[i] < min_value){
                #pragma omp critical
                min_value = std::min(min_value, scalar_image.data[i]);
            }
            if (scalar_image.data[i] > max_value){
                #pragma omp critical
                max_value =  std::max(max_value, scalar_image.data[i]);
            }
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < scalar_image.data.size(); i++){
        if (scalar_image.data[i] != INVALID_VALUE){
            scalar_image.data[i] = (scalar_image.data[i] - min_value) / (max_value - min_value);
        }
    }

}

/// <summary>
/// Converts a disparity image to a normalized depth image.
/// Ignores invalid disparity values.
/// </summary>
/// <param name="disparity">disparity in arbitrary units</param>
/// <returns>linear depth scaled from 0 to 1</returns>
ImageFloat disparityToNormalizedDepth(const ImageFloat& disparity)
{
    auto depth = ImageFloat(disparity.width, disparity.height);

    for (int i = 0; i < disparity.data.size(); i++){
        if (disparity.data[i] != INVALID_VALUE){
            depth.data[i] = 1.0 / disparity.data[i];
        }
        else{
            depth.data[i] = INVALID_VALUE;
        }
    }
        
    normalizeValidValues(depth);

    return depth;
}

/// <summary>
/// Convert linear normalized depth to target pixel disparity.
/// Invalid pixels 
/// </summary>
/// <param name="depth">Normalized depth image (values in [0,1])</param>
/// <param name="iod_mm">Inter-ocular distance in mm.</param>
/// <param name="px_size_mm">Pixel size in mm.</param>
/// <param name="screen_distance_mm">Screen distance from eyes in mm.</param>
/// <param name="near_plane_mm">Near plane distance from eyes in mm.</param>
/// <param name="far_plane_mm">Far plane distance from eyes in mm.</param>
/// <returns>screen disparity in pixels</returns>
ImageFloat normalizedDepthToDisparity(
    const ImageFloat& depth, const float iod_mm,
    const float px_size_mm, const float screen_distance_mm,
    const float near_plane_mm, const float far_plane_mm)
{
    auto px_disparity = ImageFloat(depth.width, depth.height);

    for (int y = 0; y< depth.height; y++){
        for (int x = 0; x < depth.width; x++){
            if (depth.data[y * depth.width + x] == INVALID_VALUE){

                px_disparity.data[y * depth.width + x] = INVALID_VALUE;
                continue;
            }

            float relative_depth = depth.data[y*depth.width + x] * (far_plane_mm - near_plane_mm)  - (far_plane_mm - near_plane_mm)/2; //between -(far_plane_mm - near_plane)/2 and (far_plane_mm -near_plane)/2
            float disparity_mm = iod_mm * (relative_depth / (screen_distance_mm + relative_depth)); //slides' formula
            px_disparity.data[y * depth.width + x] = disparity_mm / px_size_mm;


        }
    }

    return px_disparity; // returns disparity measured in pixels
}


/// <summary>
/// Forward-warps an image based on the given disparity and warp_factor.
/// </summary>
/// <param name="src_image">Source image.</param>
/// <param name="src_depth">Depth image used for Z-Testing</param>
/// <param name="disparity">Disparity of the source image in pixels.</param>
/// <param name="warp_factor">Multiplier of the disparity.</param>
/// <returns>ImageWithMask, containing the forward-warped image and a mask image. Mask=1 for valid pixels, Mask=0 for holes</returns>
ImageWithMask forwardWarpImage(const ImageRGB& src_image, const ImageFloat& src_depth, const ImageFloat& disparity, const float warp_factor)
{
    // The dimensions of src image, src depth and disparity maps all match.
    assert(src_image.width == disparity.width && src_image.height == disparity.height);
    assert(src_image.width == disparity.width && src_depth.height == src_depth.height);
    
    // Create a new image and depth map for the output.
    auto dst_image = ImageRGB(src_image.width, src_image.height);
    auto dst_mask = ImageFloat(src_depth.width, src_depth.height);
    // Fill the destination depth mask map with zero.
    std::fill(dst_mask.data.begin(), dst_mask.data.end(), 0.0f);
    auto dst_depth = ImageFloat(src_depth.width, src_depth.height);
    // Fill the destination depth map with a very large number.
    std::fill(dst_depth.data.begin(), dst_depth.data.end(), std::numeric_limits<float>::max());

    #pragma omp parallel for 
    for (int y = 0; y < src_image.height; y++){
            for (int x = 0; x < src_image.width; x++){

                // Compute where the pixel should be warped to based on the associated disparity and warp_factor.
                //use half-up rounding for x
                int x_prime = (int)(x + disparity.data[y * src_image.width + x] * warp_factor + 0.5f);
                int y_prime = y;

                // Check the destination depth at the [x',y'] location and compare it with the depth of the currently warped pixel.
                if (x_prime >= 0 && x_prime < src_image.width && y_prime >= 0 && y_prime < src_image.height)
                {   
                
                    float depth_prime = dst_depth.data[y_prime * src_image.width + x_prime]; 
                    float depth = src_depth.data[y * src_image.width + x];

                    if (depth < depth_prime )
                    {   
                        #pragma omp critical
                        if (depth < dst_depth.data[y_prime * src_image.width + x_prime]){ //value might have changed in the meantime

                        dst_image.data[y_prime * src_image.width + x_prime] = src_image.data[y * src_image.width + x];
                        dst_depth.data[y_prime * src_image.width + x_prime] = depth;
                        dst_mask.data[y_prime* src_image.width + x_prime] = 1.0f;
                        }
                        
                    }
                    
                }
            }
        }

    ImageWithMask imask = ImageWithMask();
    imask.mask = dst_mask;
    imask.image = dst_image;
    // Return the warped image.
    return imask;
    //return ImageWithMask(dst_image, dst_mask);

}


/// <summary>
/// Applies the bilateral filter on the given image to fill the holes
/// indicated by a binary mask (mask==0 -> missing pixel).
/// Keeps the pixels not marked as holes unchanged.
/// </summary>
/// <param name="img_forward">The image to be filtered and its mask.</param>
/// <param name="size">The kernel size. It is always odd.</param>
/// <param name="guide_sigma">Sigma value of the gaussian guide kernel.</param>
/// <returns>ImageRGB, the filtered forward warping image.</returns>
ImageRGB inpaintHoles(ImageWithMask& img, const int size)
{
    // The filter size is always odd.
    assert(size % 2 == 1);

    // Rule of thumb for gaussian's std dev.
    const float sigma = (size - 1) / 2 / 3.2f;

    // The output is initialized by copy of the input.
    auto result = ImageRGB(img.image);
    std::copy(img.image.data.begin(), img.image.data.end(), result.data.begin());
    #pragma omp parallel for shared(result) collapse(2)
        for (int y = 0; y< img.image.height; ++y){
            for (int x = 0; x < img.image.width; ++x){
                if (img.mask.data[y*img.mask.width + x] < 0.5f){

                    //result.data[y * result.width + x] = glm::vec3(1.0f);
                    //continue;

                    float sum_r = 0.0f;
                    float sum_g = 0.0f;
                    float sum_b = 0.0f;
                    float sum_weight = 0.0f;

                    for (int i = 0; i < size; ++i){
                        for (int j = 0; j < size; ++j){
                            int x0 = x - size / 2 + i;
                            int y0 = y - size / 2 + j;

                            //if the neighbor is invalid, skip it
                            if (x0 < 0 || x0 >= img.image.width || y0 < 0 || y0 >= img.image.height || img.mask.data[y0 * img.mask.width + x0] < 0.5f ){
                                continue;
                            }

                            float dist = glm::distance(glm::vec2(x, y), glm::vec2(x0, y0));
                            float w_i = gauss(dist, sigma); 

                            sum_r += w_i * img.image.data[y0 * img.image.width + x0].r;
                            sum_g += w_i * img.image.data[y0 * img.image.width + x0].g;
                            sum_b += w_i * img.image.data[y0 * img.image.width + x0].b;
                            sum_weight += w_i;
                        }
                    }
                    if (sum_weight > 0.0f){
                        result.data[y * result.width + x].r = sum_r / sum_weight;
                        result.data[y * result.width + x].g = sum_g / sum_weight;
                        result.data[y * result.width + x].b = sum_b / sum_weight;
                    }
                    else{
                        

                    }
                    
                    
                }

            }
        }

    return result;
}


