import numpy as np

class MyColor():
    '''
    COLOR   = (  B,   G,   R)
    '''
    RED     = (  0,   0, 255)
    GREEN   = (  0, 255,   0)
    BLUE    = (255,   0,   0)
    WHITE   = (255, 255, 255)
    BLACK   = (  0,   0,   0)

class MyDirection():

    TOP     = 0
    BOTTOM  = 1
    LEFT    = 2
    RIGHT   = 3

class MyQuad():
    
    def __init__(self):
        self.quad_vertexes = []

def rotation_matrix(angle):
    theta = np.radians(angle)
    np.cos(theta), np.sin(theta)
    R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))
    return R
