import math, cv2
import numpy as np

class ImageTransform:
    shape = []
    M = []

    # All calculations are done during class init
    def __init__(self, shape, src_points, dst_points):
        self.shape = shape
        src = np.float32(src_points)
        dst = np.float32(dst_points)
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)


    # Warps any image since M matrix is the same for every image
    def get_warp_image(self, image):
        warped = cv2.warpPerspective(
            image, self.M, self.shape, flags=cv2.INTER_LINEAR)
        return warped
    
    # Unwarps any image 
    def get_unwarp_image(self, image):
        unwarped = cv2.warpPerspective(
            image, self.Minv, self.shape, flags=cv2.INTER_LINEAR)
        return unwarped
    

