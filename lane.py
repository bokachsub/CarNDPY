import math, cv2
import matplotlib.pyplot as plt
import numpy as np
from line import *

class Lane:
    # region Vars
    # default field values
    nwindows = 9  # default number of sliding windows (height = 360/nwindows) 9
    margin = 300  # default width of the sliding window = 100
    minpix = 150  # default min pixel to recenter window = 50

    ploty_points = [] # [0...719] , Y values to be passed for plotting 

    window_height = 0

    shape = [] # common, since all frames are the same shape 
    midpoint = 0

    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    

    def __init__(self, img_shape):
        self.shape = img_shape
        self.ploty_points = np.linspace(0, self.shape[0]-1, self.shape[0]) # [0...1279] , Y values to be passed for plotting 
        self.midpoint = np.int(self.shape[0]//2)
        self.window_height = np.int(self.shape[0]//self.nwindows)

        self.left_fitx_points = None # actual left points taken from current coefficients
        self.right_fitx_points = None # actual right points taken from current coefficients

        self.last_good_left_fit_coefficients = None # last known good left coefficients
        self.last_good_right_fit_coefficients = None # last known good right coefficients

        self.left_fit_coefficients = None  # current known left_fit (array with polynomial coefficients)
        self.right_fit_coefficients = None # current known right_fit (array with polynomial coefficients)

        self.all_left_coefficients = [ [],[],[] ] # keeping left coefficients for mean calculation
        self.all_right_coefficients = [ [],[],[] ] # keeping right coefficients for mean calculation
        self.last_several_coeff = 25
        self.coeff_count = 0

        self.parallel_value = None

        #self.left_line = Line()
        #self.right_line = Line()

    # endregion

    # INPUT: Just an image, nothing else, everything is created from scratch
    # OUTPUT: leftx_raw, lefty_raw, rightx_raw, righty_raw RAW pixels that needed for fit_poly2
    def find_lane_pixels_extensive_search(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

        leftx_base = np.argmax(histogram[:self.midpoint])
        rightx_base = np.argmax(histogram[self.midpoint:]) + self.midpoint
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.shape[0] - (window+1)*self.window_height
            win_y_high = self.shape[0] - window*self.window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
        
            # Draw the windows on the visualization image
            #cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            #(win_xleft_high,win_y_high),(0,255,0), 2) 
            #cv2.rectangle(out_img,(win_xright_low,win_y_low),
            #(win_xright_high,win_y_high),(0,255,0), 2) 
        
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
        
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            #print(f'Good left indices in nWindow: {len(good_left_inds)}, good right indices: {len(good_right_inds)}')
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx_raw = nonzerox[left_lane_inds]
        lefty_raw = nonzeroy[left_lane_inds] 
        rightx_raw = nonzerox[right_lane_inds]
        righty_raw = nonzeroy[right_lane_inds]

        return leftx_raw, lefty_raw, rightx_raw, righty_raw
    
    # INPUT: leftx_raw, lefty_raw, rightx_raw, righty_raw RAW pixels (either from find_lane_fixels or search_around_poly)
    # INSIDE: updates coefficients, calculates mean value of coefficients
    # OUTPUT: all that needed to draw curved lines (left_fitx_points, right_fity_points)
    def fit_poly2(self, leftx_raw, lefty_raw, rightx_raw, righty_raw):       
        # saving old good coefficients if those exist
        if (not self.left_fit_coefficients is None) and (not self.right_fit_coefficients is None):
            self.last_good_left_fit_coefficients = self.left_fit_coefficients 
            self.last_good_right_fit_coefficients = self.right_fit_coefficients

        # finding new coefficients
        self.left_fit_coefficients = np.polyfit(lefty_raw, leftx_raw, 2)
        self.right_fit_coefficients = np.polyfit(righty_raw, rightx_raw, 2)        

        # finding how good those new coefficients are        
        self.parallel_value = (self.left_fit_coefficients[0] * self.right_fit_coefficients[1]) - ( self.right_fit_coefficients[0]* self.left_fit_coefficients[1])
        
        # if not that good then replacing them with old good coefficients
        if (np.absolute(self.parallel_value) > 0.00011) and (not self.last_good_left_fit_coefficients is None) and (not self.last_good_right_fit_coefficients is None):            
            self.left_fit_coefficients = self.last_good_left_fit_coefficients
            self.right_fit_coefficients = self.last_good_right_fit_coefficients        
        
        self.all_left_coefficients[0].append(self.left_fit_coefficients[0])
        self.all_left_coefficients[1].append(self.left_fit_coefficients[1])
        self.all_left_coefficients[2].append(self.left_fit_coefficients[2])        
        self.all_right_coefficients[0].append(self.right_fit_coefficients[0])
        self.all_right_coefficients[1].append(self.right_fit_coefficients[1])
        self.all_right_coefficients[2].append(self.right_fit_coefficients[2])
        self.coeff_count +=1
        
        if self.coeff_count>=self.last_several_coeff: # if there are enough coefficients then averaging last several
             lastSeveral_left_A = np.float(np.mean(self.all_left_coefficients[0][self.coeff_count-self.last_several_coeff:])) # e.g. 7 - 5 = 2 
             lastSeveral_left_B = np.float(np.mean(self.all_left_coefficients[1][self.coeff_count-self.last_several_coeff:]))
             lastSeveral_left_C = np.float(np.mean(self.all_left_coefficients[2][self.coeff_count-self.last_several_coeff:]))
 
             lastSeveral_right_A = np.float(np.mean(self.all_right_coefficients[0][self.coeff_count-self.last_several_coeff:]))
             lastSeveral_right_B = np.float(np.mean(self.all_right_coefficients[1][self.coeff_count-self.last_several_coeff:]))
             lastSeveral_right_C = np.float(np.mean(self.all_right_coefficients[2][self.coeff_count-self.last_several_coeff:]))

             left_fitx_points =  lastSeveral_left_A*self.ploty_points**2 + lastSeveral_left_B*self.ploty_points + lastSeveral_left_C
             right_fitx_points = lastSeveral_right_A*self.ploty_points**2 + lastSeveral_right_B*self.ploty_points + lastSeveral_right_C  
        else: 
            left_fitx_points =  self.left_fit_coefficients[0]*self.ploty_points**2 + self.left_fit_coefficients[1]*self.ploty_points + self.left_fit_coefficients[2]
            right_fitx_points = self.right_fit_coefficients[0]*self.ploty_points**2 + self.right_fit_coefficients[1]*self.ploty_points + self.right_fit_coefficients[2]    
        
        self.left_fitx_points = left_fitx_points
        self.right_fitx_points = right_fitx_points

        return left_fitx_points, right_fitx_points

    # measures distance from the center of the lane (between lines)
    def get_distance_from_center(self):
        current_width = self.shape[0]
        offset_from_windshield = 70
        distance_between_lines = self.right_fitx_points[current_width -1 - offset_from_windshield] - self.left_fitx_points[current_width -1 -offset_from_windshield]
        center_point_between_lines = self.left_fitx_points[current_width -1 -offset_from_windshield] + distance_between_lines/2
        distance_off_center_in_meters = (center_point_between_lines - (current_width/2) ) * self.xm_per_pix
        return round(distance_off_center_in_meters, 1)

    
    # supplies linear Y points and concatenates lists of (x,y) to create solid lines
    def get_points_for_solid_area(self, left_fitx_points, right_fitx_points):
        reversed_right_fitx_points = right_fitx_points[::-1]
        y_points = np.linspace(0, self.shape[0]-1, self.shape[0])
        y_points_reversed = y_points[::-1]
        left_zip = list(zip(left_fitx_points,y_points))
        right_zip = list(zip(right_fitx_points, y_points_reversed))
        full_list_points = left_zip + right_zip
        return np.asarray(full_list_points, dtype=np.int32)

    # INPUT: Current coefficients from class
    # OUTPUT: leftx, lefty, rightx, righty raw pixels that needed for fit_poly2
    def search_around_poly(self, binary_warped ): #starts from frame 2+     
        margin = 100 # default is 100
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
    
        left_lane_inds = ((nonzerox > (self.left_fit_coefficients[0]*(nonzeroy**2) + self.left_fit_coefficients[1]*nonzeroy + 
                        self.left_fit_coefficients[2] - margin)) & (nonzerox < (self.left_fit_coefficients[0]*(nonzeroy**2) + 
                        self.left_fit_coefficients[1]*nonzeroy + self.left_fit_coefficients[2] + margin)))

        right_lane_inds = ((nonzerox > (self.right_fit_coefficients[0]*(nonzeroy**2) + self.right_fit_coefficients[1]*nonzeroy + 
                        self.right_fit_coefficients[2] - margin)) & (nonzerox < (self.right_fit_coefficients[0]*(nonzeroy**2) + 
                        self.right_fit_coefficients[1]*nonzeroy + self.right_fit_coefficients[2] + margin)))
            
        leftx_raw = nonzerox[left_lane_inds]
        lefty_raw = nonzeroy[left_lane_inds] 
        rightx_raw = nonzerox[right_lane_inds]
        righty_raw = nonzeroy[right_lane_inds]        

        return leftx_raw, lefty_raw, rightx_raw, righty_raw

    # function for testing purposes only
    def draw_helper_lines(self, binary_warped, leftx_raw, lefty_raw, rightx_raw, righty_raw ):        
        left_fitx_points, right_fitx_points = self.fit_poly2(leftx_raw, lefty_raw, rightx_raw, righty_raw)

        margin = self.margin        
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = ((nonzerox > (self.left_fit_coefficients[0]*(nonzeroy**2) + self.left_fit_coefficients[1]*nonzeroy + 
                        self.left_fit_coefficients[2] - margin)) & (nonzerox < (self.left_fit_coefficients[0]*(nonzeroy**2) + 
                        self.left_fit_coefficients[1]*nonzeroy + self.left_fit_coefficients[2] + margin)))

        right_lane_inds = ((nonzerox > (self.right_fit_coefficients[0]*(nonzeroy**2) + self.right_fit_coefficients[1]*nonzeroy + 
                        self.right_fit_coefficients[2] - margin)) & (nonzerox < (self.right_fit_coefficients[0]*(nonzeroy**2) + 
                        self.right_fit_coefficients[1]*nonzeroy + self.right_fit_coefficients[2] + margin)))

        leftx_raw = nonzerox[left_lane_inds]
        lefty_raw = nonzeroy[left_lane_inds] 
        rightx_raw = nonzerox[right_lane_inds]
        righty_raw = nonzeroy[right_lane_inds]   

        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx_points - margin, self.ploty_points]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx_points + margin, 
                                  self.ploty_points])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx_points - margin, self.ploty_points]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx_points + margin, 
                                  self.ploty_points])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.show()
    
        # Plot the polynomial lines onto the image
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        ## End visualization steps ##

    # draw function to show left and right lanes
    def draw_lines_on_image(self, out_img, left_fitx_points, right_fitx_points):
        draw_points_left = (np.asarray([left_fitx_points, self.ploty_points]).T).astype(np.int32)
        draw_points_right = (np.asarray([right_fitx_points, self.ploty_points]).T).astype(np.int32)
        cv2.polylines(out_img, [draw_points_left], False, (255,255,255))  # args: image, points, closed, color
        cv2.polylines(out_img, [draw_points_right], False, (255,255,255))  # args: image, points, closed, color
        return out_img
    
    def combined_image_output(self, img, left_fitx_points, right_fitx_points):
        draw_points_left = (np.asarray([left_fitx_points, self.ploty_points]).T).astype(np.int32)
        draw_points_right = (np.asarray([right_fitx_points, self.ploty_points]).T).astype(np.int32)
        return img
        
    # method calculates in pixels, use measure_curvature_real to calculate in meters
    def measure_curvature_pixels(self):                
        y_eval = np.max(self.ploty_points) # Define y-value where we want radius of curvature    
        # Calculation of R_curve (radius of curvature)
        left_curverad = ((1 + (2*self.left_fit_coefficients[0]*y_eval + self.left_fit_coefficients[1])**2)**1.5) / np.absolute(2*self.left_fit_coefficients[0])
        right_curverad = ((1 + (2*self.right_fit_coefficients[0]*y_eval + self.right_fit_coefficients[1])**2)**1.5) / np.absolute(2*self.right_fit_coefficients[0])    
        return left_curverad, right_curverad
    
    # successfully calculates radius of curvature in meters
    def measure_curvature_real(self):
        lastSeveral_left_A = np.float(np.mean(self.all_left_coefficients[0][self.coeff_count-self.last_several_coeff:]))
        lastSeveral_left_B = np.float(np.mean(self.all_left_coefficients[1][self.coeff_count-self.last_several_coeff:]))        
        lastSeveral_right_A = np.float(np.mean(self.all_right_coefficients[0][self.coeff_count-self.last_several_coeff:]))
        lastSeveral_right_B = np.float(np.mean(self.all_right_coefficients[1][self.coeff_count-self.last_several_coeff:]))

        y_eval = np.max(self.ploty_points)
        # Calculation of radius of curvature in meters using average of coefficients
        left_curv_m = ((1 + (2*lastSeveral_left_A*y_eval * self.ym_per_pix + lastSeveral_left_B)**2)**1.5) / np.absolute(2*lastSeveral_left_A)
        right_curv_m = ((1 + (2*lastSeveral_right_A*y_eval * self.ym_per_pix + lastSeveral_right_B)**2)**1.5) / np.absolute(2*lastSeveral_right_A)
        return (left_curv_m + right_curv_m)/2
    