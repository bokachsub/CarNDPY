import numpy as np
import math

class Line():
    def __init__(self):        
        self.line_is_detected = False  # detected last time        
        self.last_fitx_points = [] # x values of the last n fits of the line (looke like fitx_points )        
        self.average_x_base = None  #average x values of the fitted line over the last n iterations (apprx location of X, kind of base_X)        
        self.average_fit_coefficients = None  #polynomial coefficients averaged over the last n iterations        
        self.last_fitx_coefficients = [np.array([False])]  #polynomial coefficients for the most recent fit        
        self.radius_of_curvature = None #radius of curvature of the line in some units        
        self.line_base_pos_from_center = None #distance in meters of vehicle center from the line        
        self.coefficients_diffs = np.array([0,0,0], dtype='float') #difference in fit coefficients between last and new fits        
        self.last_x_raw = None #x values for detected line pixels     
        self.last_y_raw = None  #y values for detected line pixels
        