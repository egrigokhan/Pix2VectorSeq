import math
import random
from utils import distance, vector, unitVector, rotateVector, randomGaussian
from bezier import Bezier
import matplotlib.pyplot as plt
import numpy as np

import random
import math


def drawShapeFromPoints(
    points,
    flat_to_sharp_ratio=10,
):
    #  define path array
    sets = []

    #  loop through points
    for i in range(len(points)):
        sharp = False
        #  pick whether to generate a flat or sharp dataset
        if random.random() < (1 / (flat_to_sharp_ratio + 1)):
            #  generate a sharp dataset
            sharp = True

        #   generate set
        sets.append(create_set(points[i], 30, 90, sharp=sharp))

    return sets


def drawPath(
    path
):
    #  create figure
    plt.figure()

    #  set axis limits
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    #  loop over all path curves and draw
    for i in range(len(path)):

        # ................................. Creates an iterable list from 0 to 1.
        t_points = np.arange(0, 1, 0.01)
        points = np.array(path[i])  # .... Creates an array of coordinates.
        # ......................... Returns an array of coordinates.
        curve = Bezier.Curve(t_points, points)

        plt.plot(
            curve[:, 0],    # y-coordinates.
            curve[:, 1],   # x-coordinates.
        )

    plt.grid()
    plt.show()


def transform_matrix(matrix):
    # Initialize an empty list to store the transformed matrices
    transformed_matrices = []

    # Iterate over the rows of the input matrix in steps of 2
    for i in range(0, len(matrix)):
        # Extract the necessary elements from each row
        c1, p1, c2 = matrix[i % len(matrix)]
        c3, p2, c4 = matrix[(i+1) % len(matrix)]

        # Construct the transformed matrix from the extracted elements
        transformed_matrix = [p1, c2, c3, p2]

        # Append the transformed matrix to the list of transformed matrices
        transformed_matrices.append(transformed_matrix)

    # Return the list of transformed matrices
    return transformed_matrices


def unit_vector():
    return random.random() * 360


def angled_unit_vectors(min_angle, max_angle):
    #  create a random angle between 0 and 360
    angle1 = random.random() * 360

    #  create a random angle between min_angle and max_angle
    angle_between = random.random() * (max_angle - min_angle) + min_angle

    #  create a second angle by adding the angle between to the first angle
    angle2 = (angle1 + angle_between) % 360

    return (angle1, angle2)


def create_set(p, min_angle, max_angle, sharp=False, mag=30):
    if sharp:
        u1, u2 = angled_unit_vectors(min_angle, max_angle)
        a1, a2 = mag * random.random(), mag * random.random()

        u1_vec = degree_to_unit_vector(u1)
        u2_vec = degree_to_unit_vector(u2)

        return [
            p[0], p[1], a1 * u1_vec[0], a1 *
            u1_vec[1], a2 * u2_vec[0], a2 * u2_vec[1]
        ]

    else:
        u = unit_vector()
        a1, a2 = mag * random.random(), mag * random.random()

        u_vec = degree_to_unit_vector(u)

        return [
            p[0], p[1], a1 * u_vec[0], a1 * u_vec[1], 999999, a2 * u_vec[1]
        ]


def generate_dataset(flat_to_sharp_ratio=5):
    sharp = False
    #  pick whether to generate a flat or sharp dataset
    if random.random() < (1 / (flat_to_sharp_ratio + 1)):
        #  generate a sharp dataset
        sharp = True


def format_path(path):
    formatted_path = []
    for i in range(len(path)):
        obj = path[i]

        px = obj[0]
        py = obj[1]
        c0x = obj[2]
        c0y = obj[3]
        c1x = obj[4]
        c1y = obj[5]

        ''' flat '''
        if c1x == 999999:
            ''' find c1x by
            c0y / c0x == c1y / c1x
             '''
            c1x = c0x * (c1y / c0y)

        formatted_path.append([
            (px - c0x, py - c0y),
            (px, py),
            (px + c1x, py + c1y),
        ])

    return formatted_path


def degree_to_unit_vector(degree):
    # Convert degree to radians
    radian = math.radians(degree)

    # Calculate unit vector
    unit_vector = (math.cos(radian), math.sin(radian))

    return unit_vector
