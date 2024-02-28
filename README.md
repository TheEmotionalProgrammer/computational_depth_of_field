# Computational Depth-of-Field 

This repository contains a ready-to-use framework for computing depth maps and applying these maps for many purposes

**HOW TO RUN THE APPLICATION (LINUX/MACOS):**
1. Open terminal at project directory
1. Install the required python packages with command `pip install -r requirements.txt`
1. (**Only on Linux**) Run command `sudo apt install libopencv-dev` to install opencv (needed for ken-burns mp4 video creation)
1. (**Only on MacOs**) Run command `brew install opencv` to install opencv (needed for ken-burns mp4 video creation)
1. Run command `curl -o AdelaiDepth/LeReS/Minist_Test/res101.pth https://cloudstor.aarnet.edu.au/plus/s/lTIJF4vrvHCAI31/download` to download weights of the pretrained RGB -> Depth CNN
1. Run command `source build.sh` to build the project 
1. Run command `python3 app.py` to run the application


![](project/screenshot.png)

**IMPLEMENTED FEATURES:**

**Basic features:**
1. **Load an RGB image from disk**: click on the button "Load Image" and select the image you want to open from file system. It will be displayed in the interface. [**(see implementation details)**](https://gitlab.ewi.tudelft.nl/cgv/cs4365/student-repositories/2023-2024/cs436523itamassia/-/blob/main/project/app.py?ref_type=heads#L110)
2. **Allow users to scribble depth annotations in UI**: once an image has been loaded, click on the button "Start Drawing". From now on, you can scribble annotations with your mouse on the displayed image [**(see implementation details)**](https://gitlab.ewi.tudelft.nl/cgv/cs4365/student-repositories/2023-2024/cs436523itamassia/-/blob/main/project/app.py?ref_type=heads#L180) . If you wish to change the color of annotations, just click on the "Color" button and select the new color [**(see implementation details)**](https://gitlab.ewi.tudelft.nl/cgv/cs4365/student-repositories/2023-2024/cs436523itamassia/-/blob/main/project/app.py?ref_type=heads#L198). If you wish to change the thickness of annotations, use the "brush" bar provided. Once you have finished to scribble on the image, click on "Save Image" to save the annotated image and "Save Mask" to save the mask which specifies which pixels are annotated [**(see implementation details)**](https://gitlab.ewi.tudelft.nl/cgv/cs4365/student-repositories/2023-2024/cs436523itamassia/-/blob/main/project/app.py?ref_type=heads#L222).
3. **Diffuse annotations across the image using Poisson image editing**: once you have saved the annotated image and relative mask, click on the button "Poisson". You will be asked to select a mask from file system, a correspondent annotated image, and some parameters (number of iterations and beta). Then, the result will be automatically saved in the outputs folder. The Poisson-based implemented algorithm is the **Anisotropic Diffusion**, which diffuses depth annotations across the image taking edges into account. [**(see poisson/anisotropic diffusion implementation details)**](https://gitlab.ewi.tudelft.nl/cgv/cs4365/student-repositories/2023-2024/cs436523itamassia/-/blob/main/project/poisson_and_bilateral/src/your_code_here.h?ref_type=heads#L47)
4. **Allow users to select focus depth and aperture size**: you can select a focus point and a relative aperture size after clicking on the "Bilateral" button. [**(see implementation details)**](https://gitlab.ewi.tudelft.nl/cgv/cs4365/student-repositories/2023-2024/cs436523itamassia/-/blob/main/project/app.py?ref_type=heads#L233)
5. **Simulate depth-of-field using a spatially varying cross-bilater filter**: click on the "Bilateral" button. Right after that, click on a desired focus point on the image. Then, you will be allowed to select an aperture size and a depth-map (previously generated, for instance, with poisson image editing button). The result will be automatically saved in the `project/outputs` folder. [**(see cross-bilater filter implementation details)**](https://gitlab.ewi.tudelft.nl/cgv/cs4365/student-repositories/2023-2024/cs436523itamassia/-/blob/main/project/poisson_and_bilateral/src/your_code_here.h?ref_type=heads#L154)
6. **Save and display the result**: The result is saved in the `project/outputs` folder. Click on the "Load Image" button again and select the saved result from the folder to display it in the interface.

**Extended features:**

1. **Use a pretrained RGB -> Depth CNN to supplement the depth**: click on the button "NN estimation" to call the pretrained CNN on the currently loaded image. The result will be saved in the outputs folder. [**(see implementation details)**](https://gitlab.ewi.tudelft.nl/cgv/cs4365/student-repositories/2023-2024/cs436523itamassia/-/blob/main/project/app.py?ref_type=heads#L302)
2. **Find a user-friendly way to combine predicted depth-map and user scribbles**: Once you have created your own depth-map with poisson image editing and scribbles, you can create a merged depth map with the result of the pretrained CNN clicking on the button "Fuse Maps" [**(see implementation details)**](https://gitlab.ewi.tudelft.nl/cgv/cs4365/student-repositories/2023-2024/cs436523itamassia/-/blob/main/project/app.py?ref_type=heads#L336). You will be asked to select two depth-maps and corrispective weights. The result will be saved in a folder of your choice.
3. **Implement Ken-Burns effect with depth-based parallax**: click on the button "Ken Burns". You will be asked to select a depth-map, the length of the video, number of fps, and a type of effect (forward or back and forth). Then, the process will start and the result will be saved in the outputs folder. [**(see Forward warping implementation details)**](https://gitlab.ewi.tudelft.nl/cgv/cs4365/student-repositories/2023-2024/cs436523itamassia/-/blob/main/project/ken-burns/src/your_code_here.h?ref_type=heads#L236) [**(see Ken-Burns effect implementation details)**](https://gitlab.ewi.tudelft.nl/cgv/cs4365/student-repositories/2023-2024/cs436523itamassia/-/blob/main/project/ken-burns/src/main.cpp?ref_type=heads#L44) 

**Custom features**
1. **Edge enhancement using a pretrained RGB -> edge map for better anisotropic diffusion**: click on the button "Edge Enhance" to create a new image with enhanced edges [**(see implementation details)**](https://gitlab.ewi.tudelft.nl/cgv/cs4365/student-repositories/2023-2024/cs436523itamassia/-/blob/main/project/app.py?ref_type=heads#L412). This is very useful to obtain a better anisotropic diffusion result if the original image has edges that are not marked enough.

**TEST DATA AND EXAMPLE RESULTS:**

You can find some test data in the folder `project/images` and relative results for all the steps in the folder `project/results`.



