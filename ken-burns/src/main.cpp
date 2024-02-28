#include "your_code_here.h"
#include <omp.h>

//dirs
//static const std::filesystem::path outDirPath { OUTPUT_DIR };
static const std::filesystem::path outDirPath { std::filesystem::current_path() };

static const SceneParams SceneParameters= {
    105, 35, 0, 28,
    64.0f, 0.25f, 590.0f, 550.0f, 670.0f,
    19, 0.05f, 1.0f, 30.0f, 0,
};

int main(int argc, char** argv)
{

    // Do not add any noise to the saved images.
    std::srand(unsigned(4733668));
    const float im_write_noise_level = 0.0f;

    std::chrono::steady_clock::time_point time_start, time_end;
    printOpenMPStatus();
    
    // 0. Load inputs from files. 
    ImageRGB image = ImageRGB(argv[1]);
    SceneParams scene_params = SceneParameters;
    ImageFloat linear_depth = ImageFloat(argv[2]); //the estimated depth map
    int num_frames = std::stoi(argv[3]);
    int num_sec = std::stoi(argv[4]);
    std::string effect_type = argv[5];
    
    //obtain target disparity from the linear depth
    ImageFloat target_disparity = normalizedDepthToDisparity(
        linear_depth,
        scene_params.iod_mm,
        scene_params.px_size_mm,
        scene_params.screen_distance_mm,
        scene_params.near_plane_mm,
        scene_params.far_plane_mm);
    
    //save the target disparity
    disparityToColor(target_disparity, scene_params.out_disp_min, scene_params.out_disp_max).writeToFile(outDirPath / "outputs/target_disparity.png", 1.0f, im_write_noise_level);
    
    if (effect_type == "forward")
    {

    std::vector<cv::Mat> animation_frames_cv(num_frames * num_sec, cv::Mat());
    printf("Creating the video...\n");
    #pragma omp parallel for 
    for (int i = 0; i < num_frames * num_sec; i++) {

        //print the current frame number if it is a multiple of 10

        float factor = scene_params.warp_scale - (i * 0.025f);
        ImageWithMask img = forwardWarpImage(image, linear_depth, target_disparity, factor);
        // Inpaint the holes in the forward warping image.
        ImageRGB res = inpaintHoles(img, scene_params.bilateral_size);
        //write the image to the file system
        res.writeToFile((outDirPath / ("animation" + std::to_string(i) + ".png")).string(), 1.0f, im_write_noise_level);

        cv::Mat mat = cv::imread((outDirPath / ("animation" + std::to_string(i) + ".png")).string());
        //now that we don't need it anymore, delete the image from the file system
        std::filesystem::remove(outDirPath / ("animation" + std::to_string(i) + ".png"));
        //apply a bilateral filter to the image
 
        //add the cv::Mat to the list of frames
        animation_frames_cv[i] = mat;
    }

    //create an animated mp4 video from the list of frames
    createVideo(animation_frames_cv, outDirPath / "outputs/animation.mp4", num_frames);
   
	std::cout << "All done!" << std::endl;
    }

    else if (effect_type == "back_and_forth"){

        std::vector<cv::Mat> animation_frames_cv(num_frames * num_sec, cv::Mat());
        printf("Creating the video...\n");
        
        #pragma omp parallel for 
        for (int i = 0; i < num_frames * num_sec / 2; i++) {

        //print the current frame number if it is a multiple of 10
            float factor = scene_params.warp_scale - (i * 0.025f);

            ImageWithMask img = forwardWarpImage(image, linear_depth, target_disparity, factor);
            //Inpaint the holes in the forward warping image.
            ImageRGB res = inpaintHoles(img, scene_params.bilateral_size);

            res.writeToFile((outDirPath / ("animation" + std::to_string(i) + ".png")).string(), 1.0f, im_write_noise_level);

            cv::Mat mat = cv::imread((outDirPath / ("animation" + std::to_string(i) + ".png")).string());
            std::filesystem::remove(outDirPath / ("animation" + std::to_string(i) + ".png"));
            //add the cv::Mat to the list of frames
            animation_frames_cv[i] = mat;
            animation_frames_cv[num_frames * num_sec - i - 1] = mat;
            
        }

        createVideo(animation_frames_cv, outDirPath / "outputs/animation.mp4", num_frames);
   
	    std::cout << "All done!" << std::endl;
        
    }

    return 0;
}
