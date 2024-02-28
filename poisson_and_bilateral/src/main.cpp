#include "your_code_here.h"

int main(int argc, char** argv)
{
        
    if (std::atoi(argv[1]) == 0) //cross bilateral filter
    {

    auto image = ImageRGB(argv[2]);
    auto solved_luminance = ImageFloat(argv[3]);
    auto solved_depth = crossBilateralFilter(image, solved_luminance, std::atoi(argv[4]), std::atoi(argv[5]), std::atoi(argv[6]), std::atof(argv[7]), std::atof(argv[8]));
    std::cout << "Current path is " << std::filesystem::current_path() << '\n';
    solved_depth.writeToFile("./outputs/bilateral.png");
    std::cout << "All done!" << std::endl;

    return 0;

    }
    else if (std::atoi(argv[1]) == 1) //anisotropic diffusion 
    {

    auto image = ImageRGB(argv[2]);
    auto target = ImageRGB(argv[3]);
    auto mask = ImageFloat(argv[4]);

    std::cout << "Current path is " << std::filesystem::current_path() << '\n';
    auto target_gray = rgbToLuminance(target);
    auto image_gray = rgbToLuminance(image);
    auto solved_luminance = solvePoisson(image_gray,target_gray, mask, std::atoi(argv[5]),std::atof(argv[6]));

    //print current folder
   

    solved_luminance.writeToFile("./outputs/anisotropic.png");
    std::cout << "All done!" << std::endl;

    return 0;

    }

    return 0;
}
