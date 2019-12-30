import math, cv2
import numpy as np

class ImageColorSpace:

    # Sobel based on Canny edge detection in X or Y direction
    def abs_sobel_thresh(self, img, orient='x', thresh = (0,255)):
        # BGR since image read via cv2
        thresh_min = thresh[0]
        thresh_max = thresh[1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if (orient == 'x'):
            sobel_orient = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        elif (orient == 'y'):
            sobel_orient = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        else:
            raise Exception("Orientation should be either x or y")
        abs_sobel = np.absolute(sobel_orient)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh_min) &
                      (scaled_sobel <= thresh_max)] = 1
        return binary_output

    # MAGNITUDE(absolute value) in BOTH directions
    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scale_factor = np.max(magnitude)/255
        magnitude = (magnitude/scale_factor).astype(np.uint8)
        # binary mask where mag thresholds are met
        binary_output = np.zeros_like(magnitude)
        # thresh defines pixels to exclude, (0,255) would exclude everything, would be blank
        # need to exclude 'middle' pixels and leave only contrast (lows and highs)
        binary_output[(magnitude >= mag_thresh[0]) &
                      (magnitude <= mag_thresh[1])] = 1
        return binary_output

    # edges of particular direction
    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # take the gradient in x and y separately
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        # atan2(arctan2) value, 0 is horizontal line, 1 is vertical line
        binary_output[(absgraddir >= thresh[0]) &
                      (absgraddir <= thresh[1])] = 1
        return binary_output

    # creates custom threshold of any of 3 channels in HLS or HSV
    def threshold_color_space_channel(self, img, color_space, channel, thresh=(0, 255)):
        if(color_space.upper() == 'HLS'):
            hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)  # !!
            if(channel.upper() == 'H'):
                H = hls[:, :, 0]
                binary_output = np.zeros_like(H)
                binary_output[(H > thresh[0]) & (H <= thresh[1])] = 1
                return binary_output
            elif(channel.upper() == 'L'):
                L = hls[:, :, 1]
                binary_output = np.zeros_like(L)
                binary_output[(L > thresh[0]) & (L <= thresh[1])] = 1
                return binary_output
            elif(channel.upper() == 'S'):
                S = hls[:, :, 2]
                binary_output = np.zeros_like(S)
                binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
                return binary_output
            else:
                raise Exception(
                    f'The CHANNEL {channel} does not exist in {color_space} color space')

        elif(color_space.upper() == 'HSV'):
            hls = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # !!
            if(channel.upper() == 'H'):
                H = hls[:, :, 0]
                binary_output = np.zeros_like(H)
                binary_output[(H > thresh[0]) & (H <= thresh[1])] = 1
                return binary_output
            elif(channel.upper() == 'S'):
                S = hls[:, :, 1]
                binary_output = np.zeros_like(S)
                binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
                return binary_output
            elif(channel.upper() == 'V'):
                V = hls[:, :, 2]
                binary_output = np.zeros_like(V)
                binary_output[(V > thresh[0]) & (V <= thresh[1])] = 1
                return binary_output
            else:
                raise Exception(
                    f'The CHANNEL {channel} does not exist in {color_space} color space')

        # when none of the thresholds match:
        else:
            raise Exception(f'The COLOR SPACE {color_space} is not expected')

