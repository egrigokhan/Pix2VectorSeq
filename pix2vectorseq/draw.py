import math
import random
from pix2vectorseq.utils import distance, vector, unitVector, rotateVector, randomGaussian
from pix2vectorseq.bezier import Bezier
import matplotlib.pyplot as plt
import numpy as np

def drawShapeFromPoints(
    points,
    smoothness=0.2,
    kink=1        # multiplied by $\pi$
):
    #  define path array
    cp = {}

    #  loop through points
    for i in range(len(points)):
        #  convert i_1=i-1 and i_2=i+2 to modulo len(points)
        i_1 = (i-1 - len(points)) % len(points)
        i_2 = (i+1 - len(points)) % len(points)

        #  get the points
        p = points[i]
        p1 = points[i_1]
        p2 = points[i_2]

        #  get the distance between the points
        d = distance(p1, p2)

        #  get the vector between the points
        v = vector(p1, p2)

        #  get the unit vector
        uv = unitVector(v)

        #  kink angles
        ka1 = math.pi * randomGaussian(0, kink)
        ka2 = math.pi * randomGaussian(0, kink)

        #  define cp 1 as p - smoothness * distance * unit vector
        cp1 = (p[0] - smoothness * d * rotateVector(uv, ka1)[0],
               p[1] - smoothness * d * rotateVector(uv, ka1)[1])

        #  define cp 2 as p + smoothness * distance * unit vector
        cp2 = (p[0] + smoothness * d * rotateVector(uv, ka2)[0],
               p[1] + smoothness * d * rotateVector(uv, ka2)[1])

        #  append the path
        cp[i] = [cp2, cp1]

    path = []

    #  loop through points
    for i in range(len(points)):
        #  convert i_1=i-1 and i_2=i+2 to modulo len(points)
        i_1 = (i-1 - len(points)) % len(points)
        i_2 = (i+1 - len(points)) % len(points)

        #  get the points
        p = points[i]
        p1 = points[i_1]
        p2 = points[i_2]

        # add to path
        path.append([p, cp[i][0], cp[i_2][1], p2])

    return path

def drawPath(
    path
):
  # create figure
  plt.figure()

  # loop over all path curves and draw
  for i in range(len(path)):

    t_points = np.arange(0, 1, 0.01) #................................. Creates an iterable list from 0 to 1.
    points = np.array(path[i]) #.... Creates an array of coordinates.
    curve = Bezier.Curve(t_points, points) #......................... Returns an array of coordinates.
    
    plt.plot(
      curve[:, 0],   # x-coordinates.
      curve[:, 1]    # y-coordinates.
    )

  plt.grid()
  plt.show()

