import tkinter as tk
import cv2
import numpy as np
import math
from PIL import Image, ImageTk, ImageDraw
from tkinter import colorchooser, filedialog, simpledialog
import subprocess
import sys
import os
import torch
from IPython.display import clear_output

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Depth-Aware Image Editing")

        # Create a Canvas widget
        self.canvas = tk.Canvas(self.master)

        # Create a Load Image button
        self.load_button = tk.Button(self.master, text="Load Image", command=self.load_image)
        self.load_button.config(width=10, height=3, font=("Courier", 11))

        # Create a color picker widget for the brush color
        self.color_picker = tk.Button(self.master, text="Color", command=self.choose_color, state=tk.DISABLED)
        self.color_picker.config(width=10, height=3, font=("Courier", 11))

        # Create a Scale widget for the brush thickness
        self.thickness_scale = tk.Scale(self.master, from_=1, to=10, orient=tk.HORIZONTAL, label="Brush", command=self.set_thickness, state=tk.DISABLED)
        #center the label of thickness scale
        self.thickness_scale.config(width=10, length=100, font=("Courier", 11, "bold"))

        # Create a Save Image button
        self.save_button = tk.Button(self.master, text="Save Image", command=self.save_image, state=tk.DISABLED)
        self.save_button.config(width=10, height=3, font=("Courier", 11))

        # Create a Clear Annotations button
        self.clear_button = tk.Button(self.master, text="Clear", command=self.clear_annotations, state=tk.DISABLED)
        self.clear_button.config(width=10, height=3, font=("Courier", 11))

        # Create a Save Binary Mask button
        self.save_binary_mask_button = tk.Button(self.master, text="Save Mask", command=self.save_binary_mask, state=tk.DISABLED)
        self.save_binary_mask_button.config(width=10, height=3, font=("Courier", 11))

        # Create a Start Drawing button
        self.start_drawing_button = tk.Button(self.master, text="Start Drawing", command=self.enable_drawing, state=tk.DISABLED)
        self.start_drawing_button.config(width=10, height=3, font=("Courier", 11))

        # Create an End Drawing button
        self.end_drawing_button = tk.Button(self.master, text="End Drawing", command=self.disable_drawing, state=tk.DISABLED)
        self.end_drawing_button.config(width=10, height=3, font=("Courier", 11))

        # Initialize the drawing state to disabled
        self.drawing_enabled = False

        self.apply_anisotropic_filter_button = tk.Button(self.master, text="Poisson", command=self.apply_anisotropic, state=tk.DISABLED)
        self.apply_anisotropic_filter_button.config(width=10, height=3, font=("Courier", 11))

        # Create an Apply Bilateral Filter button
        self.apply_bilateral_filter_button = tk.Button(self.master, text="Bilateral", command=self.select_focus_point, state=tk.DISABLED)
        self.apply_bilateral_filter_button.config(width=10, height=3, font=("Courier", 11))

        self.apply_nn_estimation_button = tk.Button(self.master, text="NN Estimation", command=self.apply_nn_estimation, state=tk.DISABLED)
        self.apply_nn_estimation_button.config(width=10, height=3, font=("Courier", 11))

        # Create a Create Video button
        self.create_video_button = tk.Button(self.master, text="Ken Burns", command=self.ken_burns_effect, state=tk.DISABLED)
        self.create_video_button.config(width=10, height=3, font=("Courier", 11))

        # Create a Fuse Depth Maps button
        self.fuse_depth_maps_button = tk.Button(self.master, text="Fuse Maps", command=self.double_depth_fusion, state=tk.DISABLED)
        self.fuse_depth_maps_button.config(width=10, height=3, font=("Courier", 11))

        # Create an Edge Enhance button
        self.edge_enhance_button = tk.Button(self.master, text="Edge Enhance", command=self.edge_enhance, state=tk.DISABLED)
        self.edge_enhance_button.config(width=10, height=3, font=("Courier", 11))

        # Initialize the depth focus point to None
        self.depth_focus_point = None

        #Initialize the annotated image as None
        self.annotated_img = None

        # Initialize the annotations color and thickness
        self.color = 'white'
        self.thickness = 1

        #configure the grid of buttons
        self.load_button.grid(row=1, column=0, padx=5, pady=5)
        self.start_drawing_button.grid(row=1, column=1, padx=5, pady=5)
        self.end_drawing_button.grid(row=1, column=2, padx=5, pady=5)
        self.clear_button.grid(row=1, column=3, padx=5, pady=5)
        self.color_picker.grid(row=2, column=0, padx=5, pady=5)
        self.thickness_scale.grid(row=2, column=1, padx=5, pady=5)
        self.save_button.grid(row=2, column=2, padx=5, pady=5)
        self.save_binary_mask_button.grid(row=2, column=3, padx=5, pady=5)
        self.edge_enhance_button.grid(row=3, column=0, padx=5, pady=5)
        self.apply_anisotropic_filter_button.grid(row=3, column=1, padx=5, pady=5)
        self.apply_nn_estimation_button.grid(row=3, column=2, padx=5, pady=5)
        self.fuse_depth_maps_button.grid(row=3, column=3, padx=5, pady=5)
        self.apply_bilateral_filter_button.grid(row=4, column=0, padx=5, pady=5)
        self.create_video_button.grid(row=4, column=1, padx=5, pady=5)

        for i in range(1,5):
            self.master.grid_rowconfigure(i, weight=1)
        for i in range(4):
            self.master.grid_columnconfigure(i, weight=1)
           
    def load_image(self): # Load an image file

        img_path = filedialog.askopenfilename()
        if img_path:
            self.img_path = img_path
            self.img = cv2.imread(self.img_path)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            #initialize the annotated image as a copy of the original image
            self.original_pil = Image.fromarray(self.img)
            self.annotated_img = np.array(self.original_pil)
            self.img_pil = Image.fromarray(self.img)
            self.img_tk = ImageTk.PhotoImage(self.img_pil)
            self.binary_mask = np.zeros((self.img_pil.height, self.img_pil.width), dtype=np.uint8)
            self.mask_pil = Image.fromarray(self.binary_mask)
            # Store coordinates of drawn points
            self.points = []

            # Display the image on the Canvas
            self.canvas.config(width=self.img_pil.width, height=self.img_pil.height)
            self.image_canvas = tk.Canvas(self.master, width=self.img_pil.width, height=self.img_pil.height)
            self.image_canvas.grid(row=0, column=0, columnspan = 4, rowspan = 1, padx=5, pady=5)
            self.master.grid_rowconfigure(0, weight=0)

            #grid is configured again to allow the image to be displayed correctly
            for i in range(1,5):
                self.master.grid_rowconfigure(i, weight=1)
            for i in range(4):
                self.master.grid_columnconfigure(i, weight=1)

            self.image_canvas.create_image(0, 0, image=self.img_tk, anchor = 'nw')
            self.canvas = self.image_canvas

        

            # enable the buttons
            self.color_picker.config(state=tk.NORMAL)
            self.thickness_scale.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
            self.clear_button.config(state=tk.NORMAL)
            self.start_drawing_button.config(state=tk.NORMAL)
            self.end_drawing_button.config(state=tk.NORMAL)
            self.apply_bilateral_filter_button.config(state=tk.NORMAL)
            self.save_binary_mask_button.config(state=tk.NORMAL)
            self.apply_anisotropic_filter_button.config(state=tk.NORMAL)
            self.apply_nn_estimation_button.config(state=tk.NORMAL)
            self.create_video_button.config(state=tk.NORMAL)
            self.fuse_depth_maps_button.config(state=tk.NORMAL)
            self.edge_enhance_button.config(state=tk.NORMAL)

            # Reset the depth focus point every time an image is loaded
            self.depth_focus_point = None

            # Reset the drawing state every time an image is loaded
            self.drawing_enabled = False

    def enable_drawing(self):  # Enable drawing on the image
        self.drawing_enabled = True
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)

    def disable_drawing(self): # Disable drawing on the image
        self.drawing_enabled = False
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")

    def start_draw(self, event): # Start drawing on the image
        if self.drawing_enabled:
            self.points = [(event.x, event.y)]
            

    def draw(self, event): # Draw on the image
        if self.drawing_enabled:
            # Draw lines connecting the points
            x, y = event.x, event.y
            self.points.append((x, y))
            if len(self.points) >= 2:
                # Draw a line between the last two points
                if self.color is not None:
                    self.canvas.create_line(self.points[-2], self.points[-1], fill=self.color, width=self.thickness, tags="annotation")
            draw = ImageDraw.Draw(self.img_pil) #draw on the image
            draw2 = ImageDraw.Draw(self.mask_pil) #draw on the mask
            for i in range(len(self.points)-1):
                draw.line([self.points[i], self.points[i+1]], fill=self.color, width=self.thickness)
                draw2.line([self.points[i], self.points[i+1]], fill=1, width=self.thickness)

            self.annotated_img = np.array(self.img_pil)
            self.binary_mask = np.array(self.mask_pil)

    def choose_color(self): # Display a color picker dialog box
        color = colorchooser.askcolor()[1]
        self.color = color
        return color
    
    def set_thickness(self, value): # Set the brush thickness
        self.thickness = int(value)
    
    def save_image(self): # Save the annotated image
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg")
        if save_path:
            self.annotated_img = cv2.cvtColor(self.annotated_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, self.annotated_img)
            self.annotated_img = cv2.cvtColor(self.annotated_img, cv2.COLOR_BGR2RGB)

    def clear_annotations(self): # Clear only the annotations and the stored points
    
        self.points = []
        self.binary_mask = np.zeros((self.img_pil.height, self.img_pil.width))
        self.mask_pil = Image.fromarray(self.binary_mask)
        self.canvas.delete("annotation")
        self.img_pil = self.original_pil.copy()
        self.annotated_img = np.array(self.img_pil)
    
    def save_binary_mask(self):  #save the binary mask as an image
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg")
        if save_path:
            self.binary_mask = self.binary_mask * 255 #convert the binary mask to a 0-255 image in order to save it
            cv2.imwrite(save_path, self.binary_mask)
            #restore the 0-1 values
            self.binary_mask = self.binary_mask / 255

    def select_focus_point(self): # allow the user to click on the image to select a focus point
        self.canvas.bind("<ButtonPress-1>", self.set_focus_point)

    def set_focus_point(self, event): # Set the depth focus point to the clicked pixel and select an aperture size
        self.depth_focus_point = (event.x, event.y)
        # Print the coordinates of the selected point
        print("Selected point: ({}, {})".format(event.x, event.y))
        self.canvas.unbind("<ButtonPress-1>")

        if self.depth_focus_point is None:
            # If the depth focus point has not been selected, display an error message
            tk.messagebox.showerror("Error", "Please select a depth focus point on the image.")
            return

        # Ask the user to select an aperture size
        aperture_size = simpledialog.askinteger("Aperture Size", "Enter the aperture size:")
        if not aperture_size:
            return
        
        # Apply the bilateral filter
        self.apply_bilateral_filter(aperture_size)
        self.depth_focus_point = None

    def apply_anisotropic(self): #poisson diffusion-based image editing 
        cwd = os.getcwd()
        print(cwd)
        img_path = self.img_path

        #ask the user to select the mask
        mask_path = filedialog.askopenfilename(title="SELECT THE MASK")
        if not mask_path:
            return
        
        #ask the user to select the target (annotated image)
        target_path = filedialog.askopenfilename(title="SELECT THE ANNOTATED IMAGE")
        if not target_path:
            return
        
        #ask the user to select number of iterations
        iterations = simpledialog.askinteger("Iterations", "Enter the number of iterations:")
        if not iterations:
            return
        
        #ask the user to select a value for beta. Beta is a parameter that controls the amount of diffusion
        beta = simpledialog.askfloat("Beta", "Enter the value of beta:")
        if not beta:
            return
        
        if mask_path and target_path:
            subprocess.call(args = [cwd +"/poisson_and_bilateral/build/a1_hdr",str(1),img_path,target_path, mask_path,str(iterations), str(beta)], bufsize=6)

    def apply_bilateral_filter(self, aperture_size): # Apply the cross bilateral filter on the image
        cwd = os.getcwd()
        print(cwd)

        img_path = self.img_path
        fx = self.depth_focus_point[0]
        fy = self.depth_focus_point[1]

        #ask the user for values for sigma_d and sigma_s
        sigma_d = simpledialog.askfloat("Sigma_d", "Enter the value of sigma_d:")
        if not sigma_d:
            return
        sigma_s = simpledialog.askfloat("Sigma_s", "Enter the value of sigma_s:")
        if not sigma_s:
            return
        
        #ask the user to select a depth map
        depth_pat = filedialog.askopenfilename(title="SELECT A DEPTH MAP")
        if depth_pat:
            subprocess.call(args=[cwd + "/poisson_and_bilateral/build/a1_hdr",str(0),img_path, depth_pat, str(aperture_size), str(fx), str(fy), str(sigma_d), str(sigma_s)],bufsize = 9 )

    def apply_nn_estimation(self): #depth estimation using a pretrained depth CNN

        img = cv2.imread(self.img_path) #load the image (its the one currently displayed in the app)
        
        cwd = os.getcwd() #get current working directory

        print(cwd)  #DEBUG: print it 

        cv2.imwrite(cwd + "/AdelaiDepth/LeReS/Minist_Test/test_images/choice.jpg",img) #save the image in the test_images folder

        device = torch.device('cpu') #everything runs on cpu

        os.chdir(cwd)  
        sys.path.append('/project/AdelaiDepth/Minist_Test/LeReS/')
        os.environ["PYTHONPATH"] = (":/project/AdelaiDepth/Minist_Test/LeReS")
        clear_output()

        #clean output directory Mini_Test/test_images/outputs if it exists, otherwise create it
        if os.path.exists(cwd +"/AdelaiDepth/LeReS/Minist_Test/test_images/outputs"):
            for file in os.listdir(cwd +"/AdelaiDepth/LeReS/Minist_Test/test_images/outputs"):
                os.remove(cwd +"/AdelaiDepth/LeReS/Minist_Test/test_images/outputs/"+file)
        else:
            os.mkdir(cwd +"/AdelaiDepth/LeReS/Minist_Test/test_images/outputs")

        #change directory to minist_test
        os.chdir(cwd +"/AdelaiDepth/LeReS/Minist_Test")

        #here we call the nn to process the image 
        subprocess.call(["python", "./tools/test_depth.py", "--load_ckpt", "res101.pth", "--backbone", "resnext101"])

        print("done") #DEBUG

        os.chdir(cwd) #change directory to the project folder again 

    def double_depth_fusion(self): #allow the user to select two depth maps and fuse them together in order to improve the depth estimation

        #this method can be used to fuse the depth maps obtained from the depth CNN and the depth maps obtained from the depth estimation using the anisotropic diffusion

        # Ask the user to select a first depth map
        depth_map_path = filedialog.askopenfilename(title="SELECT A DEPTH MAP")
        if not depth_map_path:
            return
        #select a weight for the first depth map
        weight1 = simpledialog.askfloat("Weight", "Enter the weight of the first depth map:")
        if not weight1:
            return
        
        # Ask the user to select a second depth map
        depth_map_path2 = filedialog.askopenfilename(title="SELECT A SECOND DEPTH MAP")
        if not depth_map_path2:
            return
        weigth2 = simpledialog.askfloat("Weight", "Enter the weight of the second depth map:")
        if not depth_map_path2:
            return
        
        #create a new depth map as a weighted average of the two selected depth maps
        depth_map1 = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
        depth_map2 = cv2.imread(depth_map_path2, cv2.IMREAD_GRAYSCALE)
        depth_map1 = depth_map1.astype(np.float32) / 255.0
        depth_map2 = depth_map2.astype(np.float32) / 255.0
        depth_map = depth_map1 * weight1 + depth_map2 * weigth2

        #save the new depth map
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg")
        if save_path:
            depth_map = depth_map * 255
            cv2.imwrite(save_path, depth_map)
            depth_map = depth_map / 255

    def ken_burns_effect(self):
        cwd = os.getcwd()
        #ask the user to select a depth map
        depth_map_path = filedialog.askopenfilename(title="SELECT A DEPTH MAP")
        if not depth_map_path:
            return
        #ask the user to select a duration for the video
        duration = simpledialog.askinteger("Duration", "Enter the duration of the video:")
        if not duration:
            return
        #ask the user to select the fps of the video. Recommended value: 60
        fps = simpledialog.askinteger("FPS", "Enter the fps of the video:")
        if not fps:
            return
            
        def ask_effect_type(): # Ask the user to select the type of effect between fixed choices
            box = tk.Toplevel()
            box.title("Effect Type")

            choice = tk.StringVar(value="forward")

            def on_submit():
                box.choice = choice.get()
                box.destroy()

            tk.Radiobutton(box, text="Forward", variable=choice, value="forward").pack(anchor=tk.W)
            tk.Radiobutton(box, text="Back and Forth", variable=choice, value="back_and_forth").pack(anchor=tk.W)
            
            tk.Button(box, text="Submit", command=on_submit).pack()

            box.wait_window(box)  # This will wait for the box to close before moving to the next line.

            return box.choice

        effect_type = ask_effect_type()
        if not effect_type:
            return
        subprocess.call(args=[cwd + "/ken-burns/build/a2_warping", self.img_path, depth_map_path, str(fps), str(duration), effect_type], bufsize = 6)

        return
        
    def edge_enhance(self): #enhance the edges of the image; this can be useful in order to improve the depth estimation using aniostropic diffusion or the cnn

        img_path = self.img_path
        img = cv2.imread(img_path)
        cv2.imwrite("./pytorch-hed/images/sample.png", img)

        cwd = os.getcwd()
        print(cwd)
        os.chdir(cwd + "/pytorch-hed")
        print(os.getcwd())

        subprocess.call(["python", "run.py", "--model", "bsds500", "--in", "./images/sample.png", "--out", "out.png"])
        
        
        #the result is in the out.png file in the pytorch-hed folder, and it is a grayscale image
        edge_img = cv2.imread("./out.png", cv2.IMREAD_GRAYSCALE)

        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) #simple sharpening kernel
        sharpened = cv2.filter2D(img, -1, kernel)
    
        mask = edge_img / 255.0
        mask3channel = cv2.merge([mask, mask, mask])
        #the image is sharpened only where the edges are
        enhanced_img = (mask3channel * sharpened + (1 - mask3channel) * img).astype(np.uint8)

        # Save the enhanced image
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg")
        if save_path:
            cv2.imwrite(save_path, enhanced_img)
        
        os.chdir(cwd)
  
  

if __name__ == "__main__":

    #start the app 
    root = tk.Tk()
    app = App(master=root)
    root.mainloop()