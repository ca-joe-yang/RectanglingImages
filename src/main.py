import cv2
import sys
import os
import numpy as np
from Image import *
from LocalWarpper import *
from GlobalWarpper import *

input_image_filename = sys.argv[1]
#output_video_filename = sys.argv[2]

def local_warp(input_image, output_video=False):

    if output_video:
        height, width, layers =  input_image.shape
        fps = 30
    
        try:
            os.remove(output_video_filename)
        except OSError:
            pass

        video = cv2.VideoWriter(output_video_filename, -1, fps, (width, height))

        for i in range(60):
            video.write(input_image.rgb)

    while not input_image.is_rect():
    #for i in range():
        try:
            canvas = input_image.get_longest_boundary_segment()
        except:
            break

        if output_video:
            for i in range(3):
                video.write(canvas)

    if output_video:
        for i in range(60):
            video.write(input_image.rgb)

def apply_grid_mesh(input_image):

    input_image.generate_grid_mesh(n_grid_x=10, n_grid_y=10)

def main():

    x = MyImage(input_image_filename)
    #local_warp_mapping = local_warp(x)
    #apply_grid_mesh(x)

    #local_warpper = LocalWarpper(input_image=x)
    #local_warp_mapping = local_warpper.run()
    
    global_warpper = GlobalWarpper(input_image=x)
    global_warpper.run()

    #c = x.draw_local_warp()
    #cv2.imshow('local_warp', c)

    cv2.waitKey(10000)
    cv2.destroyAllWindows() 

if __name__ == "__main__":
   main()
