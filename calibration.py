import math, cv2
import numpy as np
import math, sys, os, glob, cv2

class Calibration:

    # region Variables

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    objp = []  # to be initialized with zeros
    nx = 9
    ny = 6

    # params for calibration to be saved
    ret = []
    mtx = []
    dist = []
    rvecs = []
    tvecs = []
    shape = []

    # endregion

    def __init__(self, shape, nx=9, ny=6):
        self.shape = shape
        self.nx = nx
        self.ny = ny
        self.objp = np.zeros((self.nx*self.ny, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)

    def run(self, location_mask):
        # Make a list of calibration images
        images = glob.glob(location_mask)
        for fname in images:
            # print(fname)
            img = cv2.imread(fname)
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #print("Self shape: ", self.shape)
            #print("Gray shape: ", gray.shape[::-1])
            # if(gray.shape[::-1] != self.shape):
            # continue # skipping this image since shape does not match
            ret, corners = cv2.findChessboardCorners(
                gray, (self.nx, self.ny), None)  # Find the chessboard corners
            if ret == True:  # If found, draw corners
                # imgWithCorners = cv2.drawChessboardCorners(img, (self.nx,self.ny), corners, ret)
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)

        # get and fill in params for calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
        self.ret = ret
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self. tvecs = tvecs

    def get_objpoints(self):
        return self.objpoints

    def get_imgpoints(self):
        return self.imgpoints

    # long version, each time calculates parameters
    def get_undistorted_image_inline(self, image):
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
        print(gray.shape[::-1])
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        return undist

    # fast version, params taken from class fields
    def get_undistorted_image_fast(self, image, read_image=False):
        if read_image: 
            image = cv2.imread(image) 
        undist = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        return undist
