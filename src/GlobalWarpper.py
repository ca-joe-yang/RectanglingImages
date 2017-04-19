import cv2
import os
import numpy as np
from Image import MyColor

class GlobalWarpper():

    def __init__(self, input_image, orientation_n_bins=50,
                n_grid_x=10, n_grid_y=10, 
                output_video=None, record=False, record_fps=30):

        # Input
        self.input_image = input_image
        self.input_mesh = []

        # Param
        self.orientation_n_bins = orientation_n_bins

        # Output
        self.target_mesh = np.zeros((len(self.input_mesh), 1))
        self.target_rotation_angles = np.linspace(0, 180, self.orientation_n_bins+1)

    def run(self):

        dlines = self._run_lsd()
        #line_segments = self._cut_lines_into_quads(dlines)

    def _get_bin_id(self, angle):

        assert angle >= 0 and angle < 180

        for i in range(self.orientation_n_bins):
            upper_bound_angle = self.target_rotation_angles[i+1]
            if angle < upper_bound_angle:
                return i

        raise

    def _run_lsd(self):

        grey = self.input_image.grey.copy()
        lsd = cv2.createLineSegmentDetector(0, _n_bins=self.orientation_n_bins)
        dlines = lsd.detect(grey)

        canvas = self.input_image.rgb.copy()
        for dline in dlines[0]:
            x0 = int(round(dline[0][0]))
            y0 = int(round(dline[0][1]))
            x1 = int(round(dline[0][2]))
            y1 = int(round(dline[0][3]))
            cv2.line(canvas, (x0, y0), (x1, y1), MyColor.BLUE, 1)

        cv2.imshow('c', canvas)

        return dlines

    def _cut_lines_into_quads(self, dlines):
        
        line_segments = []
        for dline in dlines[0]:
            x0 = int(round(dline[0][0]))
            y0 = int(round(dline[0][1]))
            x1 = int(round(dline[0][2]))
            y1 = int(round(dline[0][3]))

