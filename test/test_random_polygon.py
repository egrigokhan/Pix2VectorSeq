import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pix2vectorseq.draw import drawShapeFromPoints, drawPath
from pix2vectorseq.bezier import Bezier
from pix2vectorseq.utils import generate_random_concave_polygon


# Example usage
N = 12
step_size = 1
angle_range = (0, 360)
num_interpolated_points = 1

points = generate_random_concave_polygon(
    N, step_size, angle_range, num_interpolated_points)
print(points)
path = drawShapeFromPoints([[100*p[0], 100*p[1]] for p in points], kink=0)

drawPath(path)
