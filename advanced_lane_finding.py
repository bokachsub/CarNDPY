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

src_points = [(274, 689), (511, 513), (780, 513),  (1047, 689)] 
dst_points = [(274, 689), (274, 513), (1047, 513), (1047, 689)] 

imageTransform = ImageTransform((1280, 720), src_points, dst_points)
imageColorSpace = ImageColorSpace()
lane = Lane((1280, 720))
visualzr = ImageVisualizer(shape=(720,1280,3))


videoEditor = VideoEditor('project_video.mp4',"project_video_output.mp4")
ret = True
while(ret == True):        
        ret, frame_distorted, frame_number = videoEditor.read_frames()  # read_frames_range(50,350)  or  read_frames()
        if frame_distorted is not None:
            print(f'Processing frame: #{frame_number}')

            frame = calib.get_undistorted_image_fast(frame_distorted)
            HLS_thresh_img = imageColorSpace.threshold_color_space_channel(frame, "HLS", "S", thresh=(170, 255)) # 170 255
            sobel_thresh_x_img = imageColorSpace.abs_sobel_thresh(frame, 'x', thresh = (20, 100))  # 20,100
            combined_HLS_S_SOBEL_X_img = np.zeros_like(sobel_thresh_x_img)
            combined_HLS_S_SOBEL_X_img[(HLS_thresh_img == 1) | (sobel_thresh_x_img == 1)] = 1
            warpedImage = imageTransform.get_warp_image(combined_HLS_S_SOBEL_X_img)

            if lane.left_fit_coefficients is None or lane.right_fit_coefficients is None :
                leftx_raw, lefty_raw, rightx_raw, righty_raw = lane.find_lane_pixels_extensive_search(warpedImage)
            else:
                leftx_raw, lefty_raw, rightx_raw, righty_raw = lane.search_around_poly(warpedImage)

            left_fitx_points, right_fitx_points = lane.fit_poly2(leftx_raw, lefty_raw, rightx_raw, righty_raw) #provides coefficients
            full_list_green_line_array = lane.get_points_for_solid_area(left_fitx_points, right_fitx_points)

            solid_image = visualzr.draw_solid_area_on_blank_image(full_list_green_line_array) #outputs solid area on blank image            
            unwarped_solid_image = imageTransform.get_unwarp_image(solid_image) # Its already colored            
            blended_image = visualzr.overlay_image_with_solid_area(frame,unwarped_solid_image)
            
            videoEditor.add_text_on_frame(blended_image,f"Radius of curvature is {round(lane.measure_curvature_real(), 1)} metres", (50,100), (255,255,255) )            
            if lane.get_distance_from_center() == 0:
                videoEditor.add_text_on_frame(blended_image,f"Vehicle is centered on the road ", (50,150), (255,255,255) )
            elif lane.get_distance_from_center()>0:
                videoEditor.add_text_on_frame(blended_image,f"Vehicle is {np.absolute(lane.get_distance_from_center())} metres to the left", (50,150), (255,255,255) )
            else:
                videoEditor.add_text_on_frame(blended_image,f"Vehicle is {np.absolute(lane.get_distance_from_center())} metres to the right", (50,150), (255,255,255) )

            #videoEditor.write_single_image_to_folder(blended_image, folder='/misc_video_frames/')         
            videoEditor.write_frame(blended_image)            
        else:
            print(f'Finished processing video')
            videoEditor.close_input_file()
            videoEditor.close_output_file()
