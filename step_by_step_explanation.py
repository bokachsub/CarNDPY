import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math, sys, os, glob, cv2
from colorama import Fore
from colorama import Style
#sys.path.append('.')
from image_visualizer import *
from calibration import *
from image_transform import *
from image_color_space import *
from lane import *
from video_editor import *

calib = Calibration((1280, 720), 9, 6)
calib.run(location_mask='camera_cal/calibration*.jpg')

src_points = [(274, 689), (511, 513), (780, 513),  (1047, 689)]  # my perfect points narrow
dst_points = [(274, 689), (274, 513), (1047, 513), (1047, 689)]  # my perfect points narrow

imageTransform = ImageTransform((1280, 720), src_points, dst_points)
imageColorSpace = ImageColorSpace()
lane = Lane((1280, 720))
visualzr = ImageVisualizer(shape=(720,1280,3))

org_img4 = cv2.imread('test_images/straight_lines1.jpg')
# ========== DISTORTION CORRECTION======================
chess1 = cv2.imread('camera_cal/calibration1.jpg')
undist_chess1 = calib.get_undistorted_image_fast('camera_cal/calibration1.jpg', read_image = True)
visualzr.plot_images([chess1, undist_chess1], ['Original image','Undistored image'])

undist_img4 = calib.get_undistorted_image_fast('test_images/straight_lines1.jpg', read_image = True)
visualzr.plot_images([org_img4, undist_img4], ['Original image','Undistored image'], invert_colors=True)

# ========== COLOR TRANSFORMS, GRADIENTS AND THRESHOLDING ============
HLS_thresh_img4 = imageColorSpace.threshold_color_space_channel(undist_img4, "HLS", "S", thresh=(170, 255)) # 170 255
sobel_thresh_x_img4 = imageColorSpace.abs_sobel_thresh(undist_img4, 'x', thresh = (20, 100))  # 20,100
combined_HLS_S_SOBEL_X_img4 = np.zeros_like(sobel_thresh_x_img4)
combined_HLS_S_SOBEL_X_img4[(HLS_thresh_img4 == 1) | (sobel_thresh_x_img4 == 1)] = 1
warpedImage4 = imageTransform.get_warp_image(combined_HLS_S_SOBEL_X_img4)

visualzr.plot_images([HLS_thresh_img4, sobel_thresh_x_img4, combined_HLS_S_SOBEL_X_img4, warpedImage4], 
['HLS_thresh_S_channel_img4','SOBEL_thresh_X_direction_img4','COMBINED_HLS_S_SOBEL_X_img4','WARPED_Image4'])

visualzr.plot_images([undist_img4, warpedImage4], ['Undistored image','Warped image'], invert_colors=True)

# =========== GETTING LINE POINTS ==========================
if lane.left_fit_coefficients is None or lane.right_fit_coefficients is None :
    leftx_raw, lefty_raw, rightx_raw, righty_raw = lane.find_lane_pixels_extensive_search(warpedImage4)
else:
    leftx_raw, lefty_raw, rightx_raw, righty_raw = lane.search_around_poly(warpedImage4)
left_fitx_points, right_fitx_points = lane.fit_poly2(leftx_raw, lefty_raw, rightx_raw, righty_raw) #provides coefficients
full_list_green_line_array4 = lane.get_points_for_solid_area(left_fitx_points, right_fitx_points)
solid_image4 = visualzr.draw_solid_area_on_blank_image(full_list_green_line_array4) #outputs solid area on blank image
lane.draw_lines_on_image(warpedImage4, left_fitx_points, right_fitx_points)
unwarped_solid_image4 = imageTransform.get_unwarp_image(solid_image4) # Its already colored!
blended_image4 = visualzr.overlay_image_with_solid_area(undist_img4,unwarped_solid_image4)
visualzr.plot_images([solid_image4, warpedImage4, blended_image4], ['Solid lane area on blank image','Lines on warped image', 'Blended image'], invert_colors=True)
