import os
import cv2
from Image import MyColor

class LocalWarpper():

    def __init__(self, input_image, 
                output_video=None, record=False, record_fps=30):

        # Input 
        self.input_image = input_image

        # Output
        self.mapping = {}

        # Record Video Setting
        self.output_video = output_video
        self.record_fps = record_fps

    def set_video_output(self, video_filename, fps=30):

        self.video_filename = video_filename
        self.video_output = True
        self.video_fps = fps

    def run(self, input_image):

        self.mapping = {}
        current_image = input_image.copy()

        def write_frames(frame, duration=1):
            for i in range(duration):
                self.video.write(frame)

        if self.video_output:
            height, width =  input_image.shape[:2]
    
            try:
                os.remove(output_video_filename)
            except OSError:
                pass

            self.video = cv2.VideoWriter(self.video_filename, -1, self.fps, (width, height))

            write_frames(current_image)

        while not current_image.is_rect():
            try:
                canvas = current_image.get_longest_boundary_segment()
                if self.video_output:
                    write_frames(canvas)
            except:
                break

        if self.video_output:
            write_frames(current_image, 60)

        return mapping
