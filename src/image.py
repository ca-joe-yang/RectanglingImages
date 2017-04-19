import cv2
import numpy as np
from util import *

class MyGridMesh():
    pass

class MyImage():

    def __init__(self, filename):
        assert filename[-4:] == '.png'
        image = cv2.imread(filename, -1)
        self.rgb_init = image[:, :, :3]
        self.mask_init = image[:, :, 3]

        self.height = image.shape[0]
        self.width = image.shape[1]

        self.seam_path_history = []
        self.local_warp_inverse_mapping = {}

        for y in range(self.height):
            for x in range(self.width):
                self.local_warp_inverse_mapping[(x, y)] = (x, y)
                if self.mask_init[y][x] == 0:
                    self.rgb_init[y][x] = 0
                else:
                    self.mask_init[y][x] = 255

        self.rgb = self.rgb_init.copy()
        self.mask = self.mask_init.copy()

        self.grey = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2GRAY)
        self._reset()

    def _reset(self):
        self._sobel_x = None
        self._sobel_y = None
        self._energy = None

    def print_detail(self):
        print('Image height: %s' % self.height)
        print('Image width: %s' % self.width)

    def show(self):
        pass
        #self.print_detail()

        #cv2.imshow('Input', self.rgb)
        #cv2.imshow('Mask', self.mask)
        #cv2.imshow('Energy', self.energy)

    def is_rect(self):
        for y in range(self.height):
            for x in range(self.width):
                if self.mask[y][x] == 0:
                    return False
        return True

    @property
    def shape(self):
        return (self.height, self.width, 3)

    @property
    def sobel_x(self):
        if self._sobel_x is None:
            self._sobel_x = cv2.Sobel(self.grey, cv2.CV_64F , 1, 0, ksize=5)
        return self._sobel_x

    @property
    def sobel_y(self):
        if self._sobel_y is None:
            self._sobel_y = cv2.Sobel(self.grey, cv2.CV_64F , 0, 1, ksize=5)
        return self._sobel_y

    @property
    def energy(self):
        if self._energy is None:
            self._energy = np.hypot(self.sobel_x, self.sobel_y)
            self._energy *= 1.0 / np.max(self._energy)
        return self._energy

    def get_longest_boundary_segment(self):
        top_len, top_start, top_end = self.get_longest_blank_segment_in_row(0)
        bottom_len, bottom_start, bottom_end = self.get_longest_blank_segment_in_row(self.height-1)
        left_len, left_start, left_end = self.get_longest_blank_seqment_in_column(0)
        right_len, right_start, right_end = self.get_longest_blank_seqment_in_column(self.width-1)

        max_len_which_border = np.argmax([top_len, bottom_len, left_len, right_len])
        #print(max_len_which_border)
        (start_pos, end_pos, x_start, x_end, y_start, y_end) = {
            0: ((top_start, 0), (top_end, 0), top_start, top_end, 0, self.height-1),
            1: ((bottom_start, self.height-1), (bottom_end, self.height-1), bottom_start, bottom_end, 0, self.height-1),
            2: ((0, left_start), (0, left_end), 0, self.width-1, left_start, left_end),
            3: ((self.width-1, right_start), (self.width-1, right_end), 0, self.width-1, right_start, right_end),
        }.get(max_len_which_border, None)

        #print(start_pos, end_pos)
        #self.mark_segment(start_pos, end_pos)
        #self.mark_domain(x_start, x_end, y_start, y_end)
        
        assert max_len_which_border in [0, 1, 2, 3]
        print(x_start, x_end, y_start, y_end)
        if max_len_which_border == 0 or max_len_which_border == 1:
            seam_path = self.find_seam_horizontal(x_start, x_end, y_start, y_end)
            # TOP : 0 --> -1, BOTTOM : 1 --> +1
            shift_direction = 2*max_len_which_border - 1
            self.add_seam_horizontal(seam_path, shift_direction=shift_direction)
        elif max_len_which_border == 2 or max_len_which_border == 3:
            seam_path = self.find_seam_vertical(x_start, x_end, y_start, y_end)
            # LEFT : 2 --> -1, RIGHT : 3 --> +1
            shift_direction = 2*max_len_which_border - 5
            self.add_seam_vertical(seam_path, shift_direction=shift_direction)

        self.seam_path_history.append(seam_path)

        #cv2.imshow('r', self.rgb)
        canvas = self.mark_paths(self.seam_path_history)
        return canvas
        #cv2.waitKey(500)

    def find_seam_horizontal(self, x_start=0, x_end=None, y_start=0, y_end=None):
        if x_end is None:
            x_end = self.width-1
        if y_end is None:
            y_end = self.height-1

        energy = self.energy[y_start:y_end+1, x_start:x_end+1]
        sub_width = x_end - x_start + 1
        sub_height = y_end - y_start + 1

        min_energy = np.zeros((sub_height, sub_width))
        min_energy_path = np.zeros((sub_height, sub_width), dtype=np.int8)
        for x in range(sub_width):
            for y in range(sub_height):
                if self.mask[y+y_start, x+x_start] == 0:
                    min_energy[y, x] = float('inf')
                    min_energy_path[y, x] = 0
                elif x == 0:
                    min_energy[y, x] = energy[y, x]
                    min_energy_path[y, x] = 0
                else:
                    if y == 0:
                        tmp = [
                            float('inf'),
                            min_energy[y, x-1],
                            min_energy[y+1, x-1],
                        ]
                    elif y == sub_height-1:
                        tmp = [
                            min_energy[y-1, x-1],
                            min_energy[y, x-1],
                            float('inf'),
                        ]
                    else:
                        tmp = [
                            min_energy[y-1, x-1],
                            min_energy[y, x-1],
                            min_energy[y+1, x-1],
                        ]
                    min_idx = np.argmin(tmp)
                    min_value = tmp[min_idx]
                    min_energy[y, x] = energy[y, x] + min_value

                    # min_idx = (0 --> -1, 1 --> 0, 2 --> +1)
                    min_energy_path[y, x] = min_idx - 1

        #print(min_energy_dir)
        seam_path = []
        for i in reversed(range(0, sub_width)):
            if i == sub_width-1:
                j = np.argmin(min_energy[:, i])
            else:
                j = j + min_energy_path[j, i+1]
            seam_path.append((j+y_start, i+x_start))
        #self.mark_path(seam_path)

        return seam_path

    def find_seam_vertical(self, x_start=0, x_end=None, y_start=0, y_end=None):
        if x_end is None:
            x_end = self.width-1
        if y_end is None:
            y_end = self.height-1

        energy = self.energy[y_start:y_end+1, x_start:x_end+1]
        sub_width = x_end - x_start + 1
        sub_height = y_end - y_start + 1

        min_energy = np.zeros((sub_height, sub_width))
        min_energy_path = np.zeros((sub_height, sub_width), dtype=np.int8)
        for y in range(sub_height):
            for x in range(sub_width):
                if self.mask[y+y_start, x+x_start] == 0:
                    min_energy[y, x] = float('inf')
                    min_energy_path[y, x] = 0
                elif y == 0:
                    min_energy[y, x] = energy[y, x]
                    min_energy_path[y, x] = 0
                else:
                    if x == 0:
                        tmp = [
                            float('inf'),
                            min_energy[y-1, x],
                            min_energy[y-1, x+1],
                        ]
                    elif x == sub_width-1:
                        tmp = [
                            min_energy[y-1, x-1],
                            min_energy[y-1, x],
                            float('inf'),
                        ]
                    else:
                        tmp = [
                            min_energy[y-1, x-1],
                            min_energy[y-1, x],
                            min_energy[y-1, x+1],
                        ]
                    min_idx = np.argmin(tmp)
                    min_value = tmp[min_idx]
                    min_energy[y, x] = energy[y, x] + min_value

                    # min_idx = (0 --> -1, 1 --> 0, 2 --> +1)
                    min_energy_path[y, x] = min_idx - 1

        #print(min_energy_dir)
        seam_path = []
        for j in reversed(range(0, sub_height)):
            if j == sub_height-1:
                i = np.argmin(min_energy[j, :])
            else:
                i = i + min_energy_path[j+1, i]
            seam_path.append((j+y_start, i+x_start))
        #self.mark_path(seam_path)

        return seam_path

    def add_seam_horizontal(self, seam_path, shift_direction):

        for pt_y, pt_x in seam_path:
            if shift_direction == 1:
                for y in reversed(range(pt_y, self.height)):
                    self.local_warp_inverse_mapping[(pt_x, y)] = self.local_warp_inverse_mapping[(pt_x, y-1)]
                for mat in [self.rgb, self.mask, self.grey]:
                    mat[pt_y+1:, pt_x] =  mat[pt_y:-1, pt_x]
                    mat[pt_y, pt_x] = mat[pt_y+1, pt_x]
            elif shift_direction == -1:
                for y in range(0, pt_y):
                    self.local_warp_inverse_mapping[(pt_x, y)] = self.local_warp_inverse_mapping[(pt_x, y+1)]
                for mat in [self.rgb, self.mask, self.grey]: 
                    mat[:pt_y, pt_x] = mat[1:pt_y+1, pt_x]
                    mat[pt_y, pt_x] = mat[pt_y-1, pt_x]

        self._reset()
        #cv2.imshow('tmp', self.rgb)

    def add_seam_vertical(self, seam_path, shift_direction):

        for pt_y, pt_x in seam_path:
            #print(pt_x, pt_y)
            if shift_direction == 1:
                for x in reversed(range(pt_x, self.width)):
                    self.local_warp_inverse_mapping[(x, pt_y)] = self.local_warp_inverse_mapping[(x-1, pt_y)]
                for mat in [self.rgb, self.mask, self.grey]:
                    mat[pt_y, pt_x+1:] = mat[pt_y, pt_x:-1]
                    mat[pt_y, pt_x] = mat[pt_y, pt_x+1]
            elif shift_direction == -1:
                for x in range(0, pt_x):
                    self.local_warp_inverse_mapping[(x, pt_y)] = self.local_warp_inverse_mapping[(x+1, pt_y)]
                for mat in [self.rgb, self.mask, self.grey]:
                    mat[pt_y, :pt_x] = mat[pt_y, 1:pt_x+1]
                    mat[pt_y, pt_x] = mat[pt_y, pt_x-1]

        self._reset()
        #cv2.imshow('tmp', self.rgb)

    def generate_grid_mesh(self, n_grid_y=5, n_grid_x=5):

        all_grid_x = np.linspace(0, self.width-1, n_grid_x, dtype=int)
        all_grid_y = np.linspace(0, self.height-1, n_grid_y, dtype=int)

        self.grid_mesh_vertices = []
        self.grid_mesh_edges = set()

        for id_y in range(n_grid_y):
            for id_x in range(n_grid_x):
                pt_1 = (all_grid_x[id_x], all_grid_y[id_y])
                self.grid_mesh_vertices.append(pt_1)
                if id_x != 0:
                    pt_2 = (all_grid_x[id_x-1], all_grid_y[id_y])
                    self.grid_mesh_edges.add((pt_1, pt_2))
                if id_x != n_grid_x-1:
                    pt_2 = (all_grid_x[id_x+1], all_grid_y[id_y])
                    self.grid_mesh_edges.add((pt_1, pt_2))
                if id_y != 0:
                    pt_2 = (all_grid_x[id_x], all_grid_y[id_y-1])
                    self.grid_mesh_edges.add((pt_1, pt_2))
                if id_y != n_grid_y-1:
                    pt_2 = (all_grid_x[id_x], all_grid_y[id_y+1])
                    self.grid_mesh_edges.add((pt_1, pt_2))

        for v in self.grid_mesh_vertices:
            #self.local_warp_inverse_mapping[v] = (new_v_x, new_v_y)
            print(v, self.local_warp_inverse_mapping[v])

        canvas = self.mark_grid_mesh(inverse=True, color=MyColor.GREEN)
        cv2.imshow('grid', canvas)

    def get_longest_blank_segment_in_row(self, row_idx):
        row = self.mask[row_idx, :]
        #if row_idx == self.height-1:
            #print(row)
        max_start, start, max_end, end = None, None, None, None
        max_length = 0
        for x in range(self.width):
            if row[x] == 0:
                if start == None:
                    start = x
                end = x
                if x != self.width - 1:
                    continue
            if start != None:
                l = end - start + 1
                if l > max_length:
                    max_length, max_start, max_end = l, start, end
                    start, end = None, None

        return max_length, max_start, max_end

    def get_longest_blank_seqment_in_column(self, column_idx):
        column = self.mask[:, column_idx]
        max_start, start, max_end, end = None, None, None, None
        max_length = 0
        for y in range(self.height):
            if column[y] == 0:
                if start == None:
                    start = y
                end = y
                if y != self.height - 1:
                    continue
            if start != None:
                l = end - start + 1
                if l > max_length:
                    max_length, max_start, max_end = l, start, end
                    start, end = None, None

        return max_length, max_start, max_end

    def mark_segment(self, left_top_pos, right_bottom_pos, color=MyColor.RED):
        canvas = self.rgb.copy()
        cv2.line(canvas, left_top_pos, right_bottom_pos, color, 1)

        return canvas

    def mark_domain(self, x_start, x_end, y_start, y_end, color=MyColor.RED):
        canvas = self.rgb.copy()
        left_top_pos = (x_start, y_start)
        right_bottom_pos = (x_end, y_end)
        cv2.rectangle(canvas, left_top_pos, right_bottom_pos, color, 1)

        return canvas

        cv2.imshow('Result', canvas)

    def mark_path(self, path, color=MyColor.RED):
        canvas = self.rgb.copy()
        
        for pt in path:
            canvas[pt[0], pt[1], :] = MyColor.RED

        return canvas

    def mark_paths(self, paths, color=MyColor.RED):
        canvas = self.rgb.copy()
        
        for path in paths:
            for pt in path:
                canvas[pt[0], pt[1], :] = MyColor.RED

        return canvas

    def mark_grid_mesh(self, inverse=False, color=MyColor.RED):
        canvas = self.rgb_init.copy()

        if inverse == False:
            for e in self.grid_mesh_edges:
                cv2.line(canvas, e[0], e[1], color, 1)
        else:
            for e in self.grid_mesh_edges:
                cv2.line(canvas, self.local_warp_inverse_mapping[e[0]], self.local_warp_inverse_mapping[e[1]], color, 1)

        return canvas
                

    def draw_local_warp(self):

        canvas = np.zeros((self.height, self.width, 3), np.uint8)

        for pt in self.local_warp_inverse_mapping:
            new_x, new_y = self.local_warp_inverse_mapping[pt]
            canvas[pt] = self.rgb_init[new_y, new_x]

        return canvas











