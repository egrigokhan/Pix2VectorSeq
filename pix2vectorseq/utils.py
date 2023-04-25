import math
import random
import random
import math
import matplotlib.pyplot as plt
import numpy as np

from cairosvg import svg2png

import pandas as pd
from tqdm import tqdm
import random

import torch

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def distance(
    p1,
    p2
):
    #  get the distance between the points
    d = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    return d


def vector(
    p1,
    p2
):
    #  get the vector between the points
    v = (p2[0] - p1[0], p2[1] - p1[1])

    return v


def unitVector(
    v
):
    #  get the unit vector
    uv = (v[0] / distance((0, 0), v), v[1] / distance((0, 0), v))

    return uv

#  a function which takes a vector and an angle and rotates the vector by the angle


def rotateVector(
    v,
    a
):
    #  get the length of the vector
    l = 1  #  distance((0, 0), v)

    #  get the angle of the vector
    angle = math.atan2(v[1], v[0])

    #  add the angle to the vector
    angle += a

    #  get the new vector
    v = (l * math.cos(angle), l * math.sin(angle))

    return v


def randomGaussian(
    mean,
    sd=1
):
    #  pick a random number from a normal distribution
    r = random.gauss(mean, sd)

    return r


""" 
    a function which will pick a random point from each quadrant of the canvas between the given bounds
"""


def randomQuadrantPoints(
    bounds,
    numPoints
):
    #  get the bounds
    xMin = bounds[0]
    xMax = bounds[1]
    yMin = bounds[2]
    yMax = bounds[3]

    #  get the width and height
    width = xMax - xMin
    height = yMax - yMin

    #  get the center
    centerX = xMin + width / 2
    centerY = yMin + height / 2

    #  get the points
    points = []
    for i in range(numPoints):
        #  pick a random quadrant
        quadrant = i

        #  pick a random point in the quadrant
        if quadrant == 0:
            x = random.randint(xMin, centerX)
            y = random.randint(yMin, centerY)
        elif quadrant == 1:
            x = random.randint(centerX, xMax)
            y = random.randint(yMin, centerY)
        elif quadrant == 3:
            x = random.randint(xMin, centerX)
            y = random.randint(centerY, yMax)
        elif quadrant == 2:
            x = random.randint(centerX, xMax)
            y = random.randint(centerY, yMax)

        #  append the point
        points.append((x, y))

    return points


def randomSegmentPoints(bounds, numSegments):
    xMin, xMax, yMin, yMax = bounds
    centerX = xMin + (xMax - xMin) / 2
    centerY = yMin + (yMax - yMin) / 2

    def random_point_in_sector(centerX, centerY, radius, angle1, angle2):
        r = random.uniform(0, radius)
        angle = random.uniform(angle1, angle2)
        x = centerX + r * math.cos(math.radians(angle))
        y = centerY + r * math.sin(math.radians(angle))
        return x, y

    points = []
    segment_angle = 360 / numSegments

    for i in range(numSegments):
        angle1 = i * segment_angle
        angle2 = (i + 1) * segment_angle
        radius = min(xMax - centerX, yMax - centerY)
        x, y = random_point_in_sector(centerX, centerY, radius, angle1, angle2)
        points.append((x, y))

    return points


def generate_random_concave_polygon(N, step_size=1, angle_range=(0, 360), num_interpolated_points=10):
    angle_min, angle_max = angle_range

    def take_step(x, y, angle, step_size):
        x_new = x + step_size * math.cos(math.radians(angle))
        y_new = y + step_size * math.sin(math.radians(angle))
        return x_new, y_new

    def cubic_bezier(t, p0, p1, p2, p3):
        return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3

    polygon_points = [(0, 0)]  # Start at the origin
    x, y = 0, 0

    for _ in range(N - 1):
        angle = random.uniform(angle_min, angle_max)
        x, y = take_step(x, y, angle, step_size)
        polygon_points.append((x, y))

    # Generate control points for Bezier curves
    control_points = [take_step(x, y, random.uniform(angle_min, angle_max), step_size)
                      for x, y in polygon_points]

    # Interpolate Bezier curves between vertices
    bezier_points = []
    for i in range(N):
        p0, p1, p2, p3 = polygon_points[i], control_points[i], control_points[(
            i + 1) % N], polygon_points[(i + 1) % N]
        for t in np.linspace(0, 1, num_interpolated_points + 1)[:-1]:
            x, y = cubic_bezier(t, p0[0], p1[0], p2[0], p3[0]), cubic_bezier(
                t, p0[1], p1[1], p2[1], p3[1])
            bezier_points.append((x, y))

    # Add the last interpolated points to close the loop
    p0, p1, p2, p3 = polygon_points[-1], control_points[-1], control_points[0], polygon_points[0]
    for t in np.linspace(0, 1, num_interpolated_points + 1):
        x, y = cubic_bezier(t, p0[0], p1[0], p2[0], p3[0]), cubic_bezier(
            t, p0[1], p1[1], p2[1], p3[1])
        bezier_points.append((x, y))

    return bezier_points


# bezier_points = generate_random_concave_polygon(N, step_size, angle_range, num_interpolated_points)
''' print("Generated Bezier Points:", bezier_points)

# Plot the generated Bezier curves
plt.figure()
x = [point[0] for point in bezier_points]
y = [point[1] for point in bezier_points]
plt.plot(x, y, marker='o')

# Example usage
N = 6
step_size = 1
angle_range = (0, 360)
num_interpolated_points = 1

points = generate_random_concave_polygon(
    N, step_size, angle_range, num_interpolated_points)
print(points)
path = drawShapeFromPoints([[100*p[0], 100*p[1]] for p in points], kink=0)
 '''

#  using this method, determine if a curve has a cusp or not from the 4 control points


def hasCusp(
    p0,
    p1,
    p2,
    p3
):
    #  get the tangent vectors
    t0 = (3 * (1 - 0) ** 2 * (p1[0] - p0[0]) + 6 * (1 - 0) * 0 * (p2[0] - p1[0]) + 3 * 0 ** 2 * (p3[0] - p2[0]),
          3 * (1 - 0) ** 2 * (p1[1] - p0[1]) + 6 * (1 - 0) * 0 * (p2[1] - p1[1]) + 3 * 0 ** 2 * (p3[1] - p2[1]))
    t1 = (3 * (1 - 1) ** 2 * (p1[0] - p0[0]) + 6 * (1 - 1) * 1 * (p2[0] - p1[0]) + 3 * 1 ** 2 * (p3[0] - p2[0]),
          3 * (1 - 1) ** 2 * (p1[1] - p0[1]) + 6 * (1 - 1) * 1 * (p2[1] - p1[1]) + 3 * 1 ** 2 * (p3[1] - p2[1]))

    #  get the slopes
    s0 = t0[1] / t0[0]
    s1 = t1[1] / t1[0]

    #  if the slopes are equal, then the curve has a cusp
    if s0 == s1:
        return True
    else:
        return False

#  turn bezier curves into svg, and color in the path


def bezierToSVG(
    curves,
    filename
):
    str_ = ""

    #  open the file
    # f = open(filename, 'w')

    #  write the header
    # f.write('<?xml version="1.0" encoding="utf-8"?>\r \r')

    #  write the svg tag
    str_ += ('<svg version="1.1" baseProfile="full" width="320" height="320" xmlns="http://www.w3.org/2000/svg">')

    #  write the path tag
    str_ += ('<path d="')

    #  loop through the curves
    for i in range(len(curves)):

        #  get the points
        p0 = curves[i][0]
        p1 = curves[i][1]
        p2 = curves[i][2]
        p3 = curves[i][3]

        #  if the curve has a cusp, then write a line
        if hasCusp(p0, p1, p2, p3):
            str_ += ('M ' + str(p0[0]) + ' ' + str(p0[1]) +
                     ' L ' + str(p3[0]) + ' ' + str(p3[1]) + ' ')
            #  otherwise, write a bezier curve
        else:
            str_ += ('M ' + str(p0[0]) + ' ' + str(p0[1]) + ' C ' + str(p1[0]) + ' ' + str(
                p1[1]) + ' ' + str(p2[0]) + ' ' + str(p2[1]) + ' ' + str(p3[0]) + ' ' + str(p3[1]) + ' ')

    #  write the path tag
    str_ = str_ + '" stroke-width="1" />'

    #  write the polygon
    str_ += (
        f'" <polygon points="{" ".join([str(x[0][0]) + "," + str(x[0][1]) for x in curves])}" fill="black" stroke="none" />')

    #  write the svg tag
    str_ += ('</svg>')

    print(str_)

    svg2png(bytestring=str_, write_to=filename)

    #  close the file
    #  f.close()


""" 
    a function which will generate a sentence from a list of points
    sentence should start with <BOS> and end with <EOS>

    example points: [[(11, 45), (19.2, 35.0), (76.4, 12.0), (87, 15)], [(87, 15), (97.6, 18.0), (72.2, 50.0), (64, 60)], [(64, 60), (55.800000000000004, 70.0), (56.6, 68.0), (46, 65)], [(46, 65), (35.4, 62.0), (2.8000000000000007, 55.0), (11, 45)], [(11, 45), (19.2, 35.0), (76.4, 12.0), (87, 15)], [(87, 15), (97.6, 18.0), (72.2, 50.0), (64, 60)], [(64, 60), (55.800000000000004, 70.0), (56.6, 68.0), (46, 65)], [(46, 65), (35.4, 62.0), (2.8000000000000007, 55.0), (11, 45)], [(11, 45), (19.2, 35.0), (76.4, 12.0), (87, 15)], [(87, 15), (97.6, 18.0), (72.2, 50.0), (64, 60)], [(64, 60), (55.800000000000004, 70.0), (56.6, 68.0), (46, 65)], [(46, 65), (35.4, 62.0), (2.8000000000000007, 55.0), (11, 45)], [(11, 45), (19.2, 35.0), (76.4, 12.0), (87, 15)], [(87, 15), (97.6, 18.0), (72.2, 50.0), (64, 60)], [(64, 60), (55.800000000000004, 70.0), (56.6, 68.0), (46, 65)], [(46, 65), (35.4, 62.0), (2.8000000000000007, 55.0), (11, 45)], [(11, 45), (19.2, 35.0), (76.4, 12.0), (87, 15)], [(87, 15), (97.6, 18.0), (72.2, 50.0), (64, 60)], [(64, 60), (55.800000000000004, 70.0), (56.6, 68.0), (46, 65)], [(46, 65), (35.4, 62.0), (2.8000000000000007, 55.0), (11, 45)], [(11, 45), (19.2, 35.0), (76.4, 12.0), (87, 15)], [(87, 15), (97.6, 18.0), (72.2, 50.0), (64, 60)], [(64, 60), (55.800000000000004, 70.0), (56.6, 68.0), (46, 65)], [(46, 65), (35.4, 62.0), (2.8000000000000007, 55.0), (11, 45)], [(11, 45), (19.2, 35.0), (76.4, 12.0), (87, 15)], [(87, 15), (97.6, 18.0), (72.2, 50.0), (64, 60)], [(64, 60), (55.800000000000004, 70.0), (56.6, 68.0), (46, 65)], [(46, 65), (35.4, 62.0), (2.8000000000000007, 55.0), (11, 45)], [(11, 45), (19.2, 35.0), (76.4, 12.0), (87, 15)], [(87, 15), (97.6, 18.0), (72.2, 50.0), (64, 60)], [(64, 60), (55.800000000000004, 70.0), (56.6, 68.0), (46, 65)], [(46, 65), (35.4, 62.0), (2.8000000000000007, 55.0), (11, 45)], [(11, 45), (19.2, 35.0), (76.4, 12.0), (87, 15)], [(87, 15), (97.6, 18.0), (72.2, 50.0), (64, 60)], [(64, 60), (55.800000000000004, 70.0), (56.6, 68.0), (46, 65)], [(46, 65), (35.4, 62.0), (2.8000000000000007, 55.0), (11, 45)], [(11, 45), (19.2, 35.0), (76.4, 12.0), (87, 15)], [(87, 15), (97.6, 18.0), (72.2, 50.0), (64, 60)], [(64, 60), (55.800000000000004, 70.0), (56.6, 68.0), (46, 65)], [(46, 65), (35.4, 62.0), (2.8000000000000007, 55.0), (11, 45)]]
 """


def generateSentence(
    points
):
    #  get the sentence
    sentence = '<BOS> <OBJ> '

    #  loop through the points
    for i in range(len(points)):
        #  get the point
        point = points[i]  #  [(11, 45), (19.2, 35.0), (76.4, 12.0), (87, 15)]

        #  get the x and y for all 4 points
        x1 = round(point[0][0], 2)
        y1 = round(point[0][1], 2)
        x2 = round(point[1][0], 2)
        y2 = round(point[1][1], 2)
        x3 = round(point[2][0], 2)
        y3 = round(point[2][1], 2)
        x4 = round(point[3][0], 2)
        y4 = round(point[3][1], 2)

        #  add the sentence
        sentence += str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + \
            ' ' + str(x3) + ' ' + str(y3) + ' ' + str(x4) + ' ' + str(y4) + ' '

    #  get the color
    color = [0, 0, 0]

    #  add color
    sentence += ' <CLR> ' + \
        str(color[0]) + ' ' + str(color[1]) + ' ' + str(color[2]) + ' '

    #  add the end of sentence token
    sentence += '<EOS>'

    #  return the sentence
    return sentence


def generateDataset(
    bounds,
    numPointsArray
):
    #  create the dataset
    dataset = pd.DataFrame(columns=['image', 'sentence'])

    #  loop through the images
    for i in tqdm(range(10000)):
        try:
            numPoints = random.choice(numPointsArray)

            #  get the points
            # randomQuadrantPoints(bounds, numPoints)
            points = randomSegmentPoints(bounds, numPoints)

            #  generate path
            path = drawShapeFromPoints(10 * points, kink=0)[:numPoints]

            #  get the image
            filename = 'images/' + str(i) + '.png'

            #  turn the curves into svg
            bezierToSVG(path, filename)

            #  get the sentence
            sentence = generateSentence(path)

            print(sentence)

            #  add the image and sentence to the dataset
            dataset = dataset.append({'image': filename, 'sentence': sentence, 'objects': [
                                     np.array(points).flatten()], 'colors': [[0, 0, 0]], 'id': i}, ignore_index=True)
        except:
            print("Error")

    #  save the dataset
    dataset.to_csv('dataset.csv', index=False)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) # device=CFG.device))
            == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(tgt, pad_idx):
    """
    tgt: shape(N, L)
    """
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(CFG.device)
    tgt_padding_mask = (tgt == pad_idx)

    return tgt_mask, tgt_padding_mask


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0]*3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
