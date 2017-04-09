import cv2
import sys
import os
import numpy as np
from image import MyImage

input_image_filename = sys.argv[1]
output_video_filename = sys.argv[2]

def local_warping(input_image):

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
        try:
            canvas = input_image.get_longest_boundary_segment()
        except:
            break

        for i in range(3):
            video.write(canvas)

    for i in range(60):
        video.write(input_image.rgb)

def apply_grid_mesh(input_image):

    height, width, layers =  input_image.shape

def main():


    x = MyImage(input_image_filename)
    local_warping(x)

    cv2.waitKey(0)
    cv2.destroyAllWindows() 

if __name__ == "__main__":
   main()
