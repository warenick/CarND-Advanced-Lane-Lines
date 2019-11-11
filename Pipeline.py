import cv2
import numpy as np
from pipeline_lib import *
from Line import Line
class Pipeline():
    def __init__(self, calibrator = None):
        self.calibrator = calibrator
        self.left_line = Line()
        self.right_line = Line()
    
    def compute_gradient(self, image, s_thresh=(170, 255), sx_thresh=(20, 100)):
        img = image.copy()
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        # Stack each channel
        binary = sxbinary | s_binary
        return binary

    def wrap(self, image, src = None, dst = None):
        img = image.copy()
        self.img_size = img.shape
        if src is None:
            src = np.float32([
                [560,450],
                [720,450],
                [1280,700],
                [0,700]])
        if dst is None:
            dst = np.float32([
                [0,0],
                [self.img_size[0],0],
                [self.img_size[0],self.img_size[1]],
                [0,self.img_size[1]]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, self.M, self.img_size , flags=cv2.INTER_LINEAR)
        return warped

    def unwrap(self, image):
        img = image.copy()
        img_size = img.shape
        mi = np.linalg.inv(self.M)
        # unwarped = cv2.warpPerspective(img, mi, self.img_size, flags=cv2.INTER_LINEAR)
        unwrapped = cv2.warpPerspective(img, mi, (self.img_size[1],self.img_size[0]))
        return unwrapped

    def draw_lane_lines(self, img,lane_left,lane_right):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        plt.imshow(result)

    def find_lines(self, img, n_windows = 9, margin = 100, minpix = 50):
        # sliding window hyperparams
        # n_windows - number of the sliding windows
        # margin - width is +/- margin        
        # minpix - min number of pixels to recenter window

        # split histogram into left and right
        # create histogram
        hist = np.sum(img[img.shape[0]//2:,:], axis=0)

        # find left and right pead on hist
        mid_p = np.int(hist.shape[0]//2)
        # first points of the lines
        leftx_base = np.argmax(hist[0:mid_p])
        rightx_base = np.argmax(hist[mid_p:]) + mid_p
        # set window height
        h_window = np.int(img.shape[0]//n_windows)
        # indentify all non zero pixels and split x and y
        nonzero = img.nonzero()
        nonzerox = np.array(nonzero[1])
        nonzeroy = np.array(nonzero[0])
        # start current pos 
        leftx_cur = leftx_base
        rightx_cur = rightx_base

        # left and right index list to receive lane pixels
        left_lane_inds = []
        right_lane_inds = []

        for window in range(n_windows):
            win_low_y = img.shape[0] - (window + 1) * h_window
            win_high_y = img.shape[0] - (window) * h_window
            win_left_low_x = leftx_cur - margin
            win_left_high_x = leftx_cur + margin
            win_right_low_x = rightx_cur - margin
            win_right_high_x = rightx_cur + margin
            
            # find nonzero indexes in windows
            good_left_inds = ((nonzeroy >= win_low_y) & (nonzeroy < win_high_y) &
                            (nonzerox >= win_left_low_x) & (nonzerox < win_left_high_x)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_low_y) & (nonzeroy < win_high_y) &
                            (nonzerox >= win_right_low_x) & (nonzerox < win_right_high_x)).nonzero()[0]

            # append inds to the lines
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # recented window if  many pixels
            if len(good_left_inds) > minpix:
                    leftx_cur = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                    rightx_cur = np.int(np.mean(nonzerox[good_right_inds]))

        # concat array of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        # find left and right pixel poses
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        return leftx, lefty, rightx, righty

    def find_poly(self, img_shape, linex, liney):
        # find coefficient for curv throught lane points
        fit = np.polyfit(liney, linex, 2) # x, y and polinome power
        # generate y points for plotting the line
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0]) # from, to, number of points
        # calculate points of  polinominals 
        try:
            polx = fit[0] * ploty**2 + fit[1] * ploty + fit[2]
        except TypeError:
            # Avoids an error if `fit` are still none or incorrect
            print('The function failed to fit a line!')
            polx = 1*ploty**2 + 1*ploty
            
        return fit, polx, ploty
    
    def draw_by_fit(self, bin_warped, left_fit, right_fit):
        # find numspace for drawing
        ploty = np.linspace(0, bin_warped.shape[0]-1, bin_warped.shape[1])
        # calculate points of polynominal
        try:
            left_polx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            right_polx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_polx = 1*ploty**2 + 1*ploty
            right_polx = 1*ploty**2 + 1*ploty
            
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(bin_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_polx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_polx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        return color_warp

    def draw_poly_bin(self, binary_warped, left_fitx, right_fitx, ploty, show_search=False, margin=100):
        
        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        result = out_img
        if show_search == True:
            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                    ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                    ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        # Plot the polynomial lines onto the image
        plt.plot(left_fitx, ploty, color='red')
        plt.plot(right_fitx, ploty, color='red')
        ## End visualization steps ##
        
        return result

    
    def search_near_poly(self, bin_img, left_polx, right_polx):
        #define margin (wide of the tube around polyline)
        margin = 80
        
        # take nonzero (ie 1 in bin image) pixels
        nonzero = bin_img.nonzero()
        nonzerox = np.array(nonzero[1])
        nonzeroy = np.array(nonzero[0])
        
        # define area for search poly +/- margin
        left_lin_inds = ((nonzerox > (left_polx[0]*(nonzeroy**2) + 
                                    left_polx[1]*nonzeroy + left_polx[2] - margin)) &
                        (nonzerox < (left_polx[0]*(nonzeroy**2) + 
                                    left_polx[1]*nonzeroy + left_polx[2] + margin)))
        right_lin_inds = ((nonzerox > (right_polx[0]*(nonzeroy**2) + 
                                    right_polx[1]*nonzeroy + right_polx[2] - margin)) &
                        (nonzerox < (right_polx[0]*(nonzeroy**2) + 
                                    right_polx[1]*nonzeroy + right_polx[2] + margin)))
        # find left and right pixel poses
        leftx = nonzerox[left_lin_inds]
        lefty = nonzeroy[left_lin_inds]
        rightx = nonzerox[right_lin_inds]
        righty = nonzeroy[right_lin_inds]
        
        return leftx, lefty, rightx, righty

    def measure_curv(self, img_shape, x, y):
        # define pixel to meters conversion coefficienst
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        # find poly coef in meters
        fit_meters = np.polyfit(y*ym_per_pix, x*xm_per_pix, 2)
        # find lin space for polynominal
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
        # set point there we want to measure curvature
        # It is bottom of the image because there is our camera on birdview
        y_eval = np.max(ploty)
        # convert eval point form pixels to meters
        y_eval_m = y_eval * ym_per_pix
        
        curvrad = ((1 + (2*fit_meters[0]*y_eval_m + fit_meters[1])**2)**1.5) / np.abs(2*fit_meters[0])

        return curvrad


    def measure_pose(self, img, left_fit, right_fit, ploty): 
        # calculate
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        car_pos = img.shape[1] / 2 * xm_per_pix
        # It is bottom of the image because there is our camera on birdview
        y_eval = np.max(ploty)
        # calculate center of the lane
        left_fpoint = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
        right_fpoint = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
        #calculate car pose
        lane_center = (right_fpoint + left_fpoint) / 2 * xm_per_pix
        pose = lane_center - car_pos            
        return pose

    def proccess(self,image):

        img = image.copy()
        undist = self.calibrator.undistort(img)
        # found gradient
        gradient = self.compute_gradient(undist)
        # wrap imgs
        wrapped = self.wrap(gradient)
        # find left and right pixel poses
        # find left and right polyline
        self.left_line.count_check()
        self.right_line.count_check()
        # if we have not lines use sliding window
        # if left_line.detected is False or right_line.detected is False:
            # find points of lines by sliding window
        lx, ly, rx, ry = self.find_lines(wrapped)
        # if we already have lines from previous image 
        # elif left_line.detected is True and right_line.detected is True:
        #     # use search near poly
        #     lx, ly, rx, ry = pipeline.search_near_poly(wrapped, left_line.current_fit, right_line.current_fit)

        # if find points of the right line
        if len(rx) > 0 and len(ry) > 0:
            # set was detected
            self.right_line.was_detected = True
            # add founded points to instanse
            self.right_line.allx = rx
            self.right_line.ally = ry
            # add poly 
            l_fit, self.right_line.polx, self.right_line.ploty = self.find_poly(wrapped.shape,self.right_line.allx, self.right_line.ally)
            self.right_line.add_fit(l_fit)
        else:
            self.right_line.count += 1 
        # if find points of left line
        if len(lx) > 0 and len(ly) > 0:
            # set was detected
            self.left_line.was_detected = True
            # add founded points to instanse
            self.left_line.allx = lx
            self.left_line.ally = ly
            # add poly 
            r_fit, self.left_line.polx, self.left_line.ploty = self.find_poly(wrapped.shape,self.left_line.allx, self.left_line.ally)
            self.left_line.add_fit(r_fit)
        else:
            self.left_line.count += 1    

        # show_img(pipeline.draw_poly_bin(wrapped, l_polx, r_polx, r_poly))
        left_curvd = self.measure_curv(wrapped.shape, lx, ly)
        right_curvd = self.measure_curv(wrapped.shape, rx, ry)
        avg_curvrd = round(np.mean([left_curvd,right_curvd]), 0)
        curv_text = f"current radius of  curvature is {avg_curvrd} meters"
        car_pose = self.measure_pose(wrapped,self.left_line.current_fit, self.right_line.current_fit, self.left_line.ploty)
        if car_pose > 0:
            pose_text = f"car is {round(abs(car_pose), 2)} meters left from center"
        else:
            pose_text = f"car is {round(abs(car_pose), 2)} meters right from center"
        # print test to image   
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(undist, curv_text, (10,100), font, 2, (30,240,30), 3)
        cv2.putText(undist, pose_text, (10,150), font, 2, (30,240,30), 3)
        #Draving lines
        # find best fits  for drawing
        self.left_line.find_best_fit()
        self.right_line.find_best_fit()
        # invert matrix of tranfer
        color_wrap = self.draw_by_fit(wrapped, self.left_line.best_fit, self.right_line.best_fit)
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        unwarp = self.unwrap(color_wrap)
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, unwarp, 0.3, 0) 
        return result