
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

I tried to create separate classes for separate purposes (camera calibration, image distortion, color thresholding, getting image warped and unwarped, visualizing results and so on) so that it would be easy to decouple the pipeline if corrections need to be made or for reuse in other projects. 

[//]: # (Image References)

[chess_undistored]: ./output_images/Chess_undistored.png "Chessboard undistorted"
[image_undistored]: ./output_images/Undistorted_image_2.png "Undistorted"
[image_transforms]: ./output_images/Transforms_gradients_thresholding_2.png "Transforms"
[image_warped]: ./output_images/Warping_image_SL1.png "Warped Image"
[image_lines]: ./output_images/Lines_with_blending_2.png "Image Lines"

[example_output]: ./output_images/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.   

This file is a writeup that describes the steps taken to achieve the desired outcome. 

Step by step instructions can be found in step_by_step_explanation.py file that is intended to work with images, not video.  

Video processing can be done with advanced_lane_finding.py file.


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

It all starts with camera calibration. I created a separate class called Calibration and placed it in `calibration.py` file. The constructor of the class accepts shape of an image and dimentions of the chessboard to calibrate upon (these params are the required for calibration and stay the same throughout the lifespan of Calibration class). Calibration can be executed by instantiation of `calibration` object and running `calibration.run()` method that accepts masked path to images. Inside the function there are two openCV calls: `cv2.findChessboardCorners()` and `cv2.calibrateCamera()` that result in output of distortion matrix and vectors for image correction. Now I have everything ready to calibrate each and every image by running `calibration.get_undistorted_image()` method. 

![Undistored image][chess_undistored]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

![Undistored image][image_undistored]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

To perform image manipulations, including thresholding image color channels and taking Sobel thresholds I created a separate class `ImageColorSpace` inside `image_color_space.py` file. Functions of the class include the following: 

| Function                           | Description   | 
|:----------------------------------:|:-------------:| 
| abs_sobel_thresh                   | Creates binary image using Sobel threshold in X or Y direction between thresh_min and thresh_max values | 
| mag_thresh                         | Creates binary image using Sobel threshold in both directions and implifying contrast pixels |
| dir_threshold                      | Creates binary image using Sobel threshold in X and Y directions and taking abs value of gradient direction |
| threshold_color_space_channel      | Creates custom threshold of any channel in HLS or HSV and outputs binary image |

In my pipeline I am performing Sobel threshold in X direction via `abs_sobel_thresh` function and thresholding `S` channel in `HLS` colorspace via `threshold_color_space_channel` function and then combining the result to get the optimized binary image. 

![Image Transforms][image_transforms]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

For image transform I have a separate class inside `image_transform.py`. Class constructor accepts shape of the image, as well as source and distination points. It has 2 functions to warp and unward images: `get_warp_image` and `get_unwarp_image`. Both of them use `cv2.warpPerspective` function with `cv2.INTER_LINEAR` flag for image transformation. 

For source and destination points I took the undistored image and selected the following points: 

| Source        | Destination   | 
|:-------------:|:-------------:| 
| (274, 689)    | (274, 689)    | 
| (511, 513)    | (274, 513)    |
| (780, 513)    | (1047, 513)   |
| (1047, 689)   | (1047, 689)   |

I verified that my perspective transform was working as expected by taking an image with highway lane that has no curve and doing a warp image transform. 

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Warped image][image_warped]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

For lane and left/right lines identification and processing I created a class `Lane` inside the `lane.py` file. This class performs all actions and transformations that has to do with lane processing. For better readability and to be able to maintain the code I am using the following naming convensions:

| Variables        | Meaning   | 
|:-------------:|:-------------:| 
| leftx_raw, rightx_raw      | Raw (chaotic, unordered) points in a binary image as a result of applying Sobel and color threshold filters    | 
| left_fit_coefficients, right_fit_coefficients  | Coefficients A,B and C in `Ax**2 + Bx + C` equation found from raw points for left and for right. |
| left_fitx_points, right_fitx_points | These are actual left and right line points (that fit) to be drawn on a final processed image |

Class `Lane` has several key functions:
* `find_lane_pixels_extensive_search()` searches for left and right points very widely, through the entire bottom half of an image (initial search)
* `search_around_poly()` is more efficient function, searches only around identified left and right lines
* `fit_poly2()` uses raw points found in one of previous 2 functions, finds coefficients and outputs real points that can be drawn on image
* `get_distance_from_center()` calculates distance from the center of the lane (how far the car is to the left or right)
* `measure_curvature_real()` calculates lane curvature in meters


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Function `measure_curvature_real()` of class `Lane` is responsible for calculations of curvature. I am taking the averaged (over the last several iterations) coefficients and putting them into the formula to get left and right lines curvatures and then averaging the result for better accuracy: 

```python
left_curv_m = ((1 + (2*lastSeveral_left_A*y_eval * self.ym_per_pix + lastSeveral_left_B)**2)**1.5) / np.absolute(2*lastSeveral_left_A)
right_curv_m = ((1 + (2*lastSeveral_right_A*y_eval * self.ym_per_pix + lastSeveral_right_B)**2)**1.5) / np.absolute(2*lastSeveral_right_A)
return (left_curv_m + right_curv_m)/2
```

I implemented `get_distance_from_center()` in the same class. I am locating `distance_between_lines` and `center_point_between_lines`. Then using `xm_per_pix` value I am performing distance off center calculation: 
```pytond
current_width = self.shape[0]
offset_from_windshield = 70
distance_between_lines = self.right_fitx_points[current_width -1 - offset_from_windshield] - self.left_fitx_points[current_width -1 -offset_from_windshield]
center_point_between_lines = self.left_fitx_points[current_width -1 -offset_from_windshield] + distance_between_lines/2
distance_off_center_in_meters = (center_point_between_lines - (current_width/2) ) * self.xm_per_pix
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.


![Example output][example_output]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the first issues that I faced was that I needed to experiment a lot and adjust parameters all the time. To make the process more productive I created a utility class ImageVisualizer and added several helper functions to ImageTransform, ImageColorSpace and Lane classes. When processing video I was also doing output of individual frames to the directory so that it is easier to track it frame by frame and compare results of different runs. 

I also tried my best to make individual image processing classes decoupled and independent from each other so that it would be easier to reuse the code or adjust params as needed. 

My algorithm assumes that lane lines are not moving radically from side to side. If this is not the case then it will most likely fail. 

I also found it challenging to track the lines with further distance from the car. It is much easier to threshold and identify lines just in front of the car compared to the lines several hundred meters away from it. 

Project also has some minor hardcoded values, which can be easily modified if needed. 
