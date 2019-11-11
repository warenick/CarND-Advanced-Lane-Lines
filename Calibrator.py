import os
import matplotlib.image as mpimg
import numpy as np
import cv2
from array import array

class Calibrator():

    def __init__(self, folder = None):
        self.imgs = []
        self.folder = folder
        self.mtx = None
        self.dist = None

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def load_calibration_images(self, folder = None):
        self.imgs = []
        if folder is None:
            folder = self.folder
        self.folder = folder

        list_files = os.listdir(folder)
        for file in list_files:
            self.imgs.append(mpimg.imread(folder+file))

    def calculate(self, nx = 9 , ny = 6):
        ret_list = []
        corners_list = []
        objp = []
        objp = np.zeros((nx*ny,3),np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
        
        imgpoints = []
        objpoints = [] 
        
        for img in self.imgs:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            # If found, draw corners
            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)
        cal_img_size = (self.imgs[0].shape[1], self.imgs[0].shape[0])
        ret, self.mtx, self.dist, rvecs, tvecs =  cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

            
    
    def save_calibration_data(self, folder = None):
        if folder is None:
            folder = self.folder
        self.folder = folder
        # TODO:realize this
        pass

    def load_calibration_data(self, folder = None):
        # if folder is None:
        #     folder = self.folder
        # self.folder = folder
        # TODO:realize this
        # self.mtx[0][0] = 1157.77942161
        # self.mtx[0][1] = 0.0
        # self.mtx[0][2] = 667.11104971
        # self.mtx[1][0] = 0.0
        # self.mtx[1][1] = 1152.82305152
        # self.mtx[1][2] = 386.1290685
        # self.mtx[2][0] = 0.0
        # self.mtx[2][1] = 0.0
        # self.mtx[2][2] = 1.0
        # self.dist[0] = -0.24688832643497355
        # self.dist[1] = -0.02372816174583693
        # self.dist[2] = -0.0010984299495592448
        # self.dist[3] = 0.00035105289714327614
        # self.dist[4] = -0.002591348498087298
        # self.mtx[0][0] [[1157.77942161, 0.0, 667.11104971], [0.0, 1152.82305152, 386.1290685],[0.0, 0.0, 1.0]])
        # self.dist = array('d',[-0.24688832643497355, -0.02372816174583693, -0.0010984299495592448, 0.00035105289714327614, -0.002591348498087298])
        pass
