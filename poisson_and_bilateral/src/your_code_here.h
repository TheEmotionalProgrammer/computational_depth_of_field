#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <span>
#include <tuple>
#include <vector>

#include "helpers.h"

/*
 * Utility functions.
 */

template<typename T>
int getImageOffset(const Image<T>& image, int x, int y)
{
    // Return offset of the pixel at column x and row y in memory such that 
    // the pixel can be accessed by image.data[offset].
    return y*image.width + x;
}

//FUNCTION USED TO CONVERT RGB TO GRAYSCALE
ImageFloat rgbToLuminance(const ImageRGB& rgb)
{
    // RGB to luminance weights defined in ITU R-REC-BT.601 in the R,G,B order.
    const auto WEIGHTS_RGB_TO_LUM = glm::vec3(0.299f, 0.587f, 0.114f);
    // An empty luminance image.
    auto luminance = ImageFloat(rgb.width, rgb.height);
    // Fill the image by logarithmic luminace.
    // Luminance is a linear combination of the red, green and blue channels using the weights above.

    for (int y = 0; y < rgb.height; y++) {
        for (int x = 0; x < rgb.width; x++) {
            int offset = getImageOffset(rgb, x, y);
            auto pixel = rgb.data[offset];
            auto luminance_pixel = glm::dot(pixel, WEIGHTS_RGB_TO_LUM);
            luminance.data[offset] = luminance_pixel;
        }
    }

    return luminance;
}

//ANISOTROPIC DIFFUSION
ImageFloat solvePoisson(const ImageFloat& grayscale_src_image, const ImageFloat& grayscale_annotated_image, const ImageFloat& mask, const int num_iters = 10000, const double beta = 40)
{
    auto I = ImageFloat(grayscale_src_image.width, grayscale_src_image.height);
    std::fill(I.data.begin(), I.data.end(), 0.0f);
    //fill I as a copy of grayscale_annotated_image
    std::copy(grayscale_src_image.data.begin(), grayscale_src_image.data.end(), I.data.begin());

    auto I_next = ImageFloat(grayscale_src_image.width, grayscale_src_image.height);
    // create I_next as a copy of I
    
    for (auto iter = 0; iter < num_iters; iter++){
        if (iter % 500 == 0) {
            // Print progress info every 500 iteartions.
            std::cout << "[" << iter << "/" << num_iters << "] Solving Poisson equation..." << std::endl;
        }

        #pragma omp parallel for shared(I, I_next) collapse(2)
        for (auto y = 0; y < grayscale_src_image.height; y++) {
            for (auto x = 0; x < grayscale_src_image.width; x++) {
            
                double first_term =0;
                if (x > 0) {
                    int offset = getImageOffset(I, x - 1, y);
                    double pixel = I.data[offset];
                    first_term = pixel;
                    
                }

                double second_term = 0;
                if (x < I.width - 1) {

                    int offset = getImageOffset(I, x + 1, y);
                    double pixel = I.data[offset];
                    second_term = pixel;

                }

                double third_term = 0;
                if (y > 0) {

                    int offset = getImageOffset(I, x, y - 1);
                    double pixel = I.data[offset];
                    third_term = pixel;
                    
                }


                double fourth_term = 0;
                if (y < I.height - 1) {

                    int offset = getImageOffset(I, x, y + 1);
                    double pixel = I.data[offset];
                    fourth_term = pixel;
                    
                }
                if (mask.data[getImageOffset(mask, x, y)] > 0.5) {
                    I_next.data[getImageOffset(I_next, x, y)] = grayscale_annotated_image.data[getImageOffset(grayscale_annotated_image, x, y)];
                }
                else{
                double first = 0;
                double second = 0;
                double third = 0;
                double fourth = 0;

                if (x > 0) {
                    first = grayscale_src_image.data[getImageOffset(grayscale_src_image, x - 1, y)];
                }
                if (x < I.width - 1) {
                    second = grayscale_src_image.data[getImageOffset(grayscale_src_image, x + 1, y)];
                }
                if (y > 0) {
                    third = grayscale_src_image.data[getImageOffset(grayscale_src_image, x, y - 1)];
                }
                if (y < I.height - 1) {
                    fourth = grayscale_src_image.data[getImageOffset(grayscale_src_image, x, y + 1)];
                }

                auto wk1 = exp(-beta * abs(grayscale_src_image.data[getImageOffset(grayscale_src_image,x,y)] - first));
                auto wk2 = exp(-beta * abs(grayscale_src_image.data[getImageOffset(grayscale_src_image, x, y)] - second));
                auto wk3 = exp(-beta * abs(grayscale_src_image.data[getImageOffset(grayscale_src_image, x, y)] - third));
                auto wk4 = exp(-beta * abs(grayscale_src_image.data[getImageOffset(grayscale_src_image, x, y)] - fourth));
                //
                // //compute the numerator and denominator of the equation
                auto numerator = (wk1 * first_term) + (wk2 * second_term) + (wk3 * third_term) + (wk4 * fourth_term);
                auto denominator = wk1 + wk2 + wk3 + wk4;
                //check if denominator is 0, if so set the pixel value to 0

                // //set the pixel value to the numerator divided by the denominator
                I_next.data[getImageOffset(I_next, x, y)] = numerator / denominator;

                }
                
        }
        }
        // Swaps the current and next solution so that the next iteration
        // uses the new solution as input and the previous solution as output.

        std::swap(I, I_next);
    }

    // After the last "swap", I is the latest solution.
    //print current folder

    return I;
}

//CROSS BILATERAL FILTER (DEPTH BASED)
ImageRGB crossBilateralFilter(const ImageRGB& img, const ImageFloat& depth_map, int aperture_size, int fx, int fy, double sigmadepth, double sigmaspace){
    auto result = ImageRGB(img.width, img.height);
    float focus_depth = 0.0f;
    focus_depth = depth_map.data[getImageOffset(depth_map, fx, fy)];
    for (auto y = 0; y < img.height; y++) {
        for (auto x = 0; x < img.width; x++) {
            float sumr = 0.0f;
            float sumg = 0.0f;
            float sumb = 0.0f;
            float sum2 = 0.0f;
            for (auto i = 0; i < aperture_size; i++) {
                for (auto j = 0; j < aperture_size; j++) {
                    int offset = getImageOffset(img, x + i - aperture_size / 2, y + j - aperture_size / 2);

                    if (offset < 0 || offset >= img.data.size()) {
                        continue;
                    }

                    if (i == aperture_size / 2 && j == aperture_size / 2) {
                        auto gi2 = exp(-((i - aperture_size / 2) * (i - aperture_size / 2) + (j - aperture_size / 2) * (j - aperture_size / 2)) / (2 * sigmaspace * sigmaspace));
                        sumr += img.data[offset].r * gi2;
                        sumg += img.data[offset].g* gi2;
                        sumb += img.data[offset].b* gi2;
                        sum2 += gi2;
                        continue;
                    }

                    auto pixel = img.data[offset];
                    auto pixel_depth = depth_map.data[offset];

                    //now we compute the gaussian for the depth distance from the focus depth
                    auto gi = pow((1 -exp(-((pixel_depth - focus_depth) * (pixel_depth - focus_depth)) / (2 * sigmadepth * sigmadepth))),2);

                    //and the gaussian for the spatial distance
                    auto gi2 = exp(-((i - aperture_size / 2) * (i - aperture_size / 2) + (j - aperture_size / 2) * (j - aperture_size / 2)) / (2 * sigmaspace * sigmaspace));

                    sumr += gi * gi2 * pixel.r;
                    sumg += gi * gi2 * pixel.g;
                    sumb += gi * gi2 * pixel.b;
                    sum2 += gi * gi2;
                }
            }
            result.data[getImageOffset(result, x, y)] = glm::vec3(sumr / sum2, sumg / sum2, sumb / sum2);
        }
    }
    return result;

}
