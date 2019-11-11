import numpy as np
class Line():
    def __init__(self):
        # counter for false detection
        self.count = 0
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # list of last 5 poly coefficients
        self.last_five_fits = []  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        # current curv pounts
        self.polx = None
        # current ploty
        self.ploty = None
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None  
   
    def add_fit(self, fit):
        self.current_fit = fit
        if len(self.last_five_fits) < 5:
            self.last_five_fits.append(fit)
        elif len(self.last_five_fits) == 5:
            self.last_five_fits.pop(0)
            self.last_five_fits.append(fit)
    
    # counter for false detection
    # if more, restart the line
    def count_check(self):
        if self.count > 5:
            self.restart()
            
            
    def find_best_fit(self):
        self.best_fit = np.mean(self.last_five_fits, axis=0)
        return self.best_fit
