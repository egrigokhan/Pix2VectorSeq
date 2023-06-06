import random
import math


def transform_matrix_from_old_to_new(matrix):
    # Initialize an empty list to store the transformed matrices
    transformed_matrices = []

    # Iterate over the input matrix
    for i in range(len(matrix)):
        # Extract the necessary elements from the current transformed matrix
        p2, c4, c5, p3 = matrix[(i - 1) % len(matrix)]
        p3, c6, c7, p4 = matrix[i % len(matrix)]

        # Construct the transformed row from the extracted elements
        transformed_row = [c5, p3, c6]

        # Append the transformed row to the list of transformed matrices
        transformed_matrices.append(transformed_row)

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


def create_set(p, min_angle, max_angle, sharp=False, mag=2):
    if sharp:
        u1, u2 = angled_unit_vectors(min_angle, max_angle)
        a1, a2 = mag * random.random(), mag * random.random()

        return [
            p, (a1, a2), u1, u2
        ]

    else:
        u = unit_vector()
        a1, a2 = mag * random.random(), mag * random.random()

        return [
            p, (a1, a2), u
        ]


def generate_dataset(flat_to_sharp_ratio=5):
    sharp = False
    #  pick whether to generate a flat or sharp dataset
    if random.random() < (1 / (flat_to_sharp_ratio + 1)):
        #  generate a sharp dataset
        sharp = True

def drawShapeFromPoints_new(
    points,
    flat_to_sharp_ratio=1,
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
         
         #  generate set
         sets.append(create_set(points[i], 0, 180, sharp=sharp))

    return sets

def format_path(path):
  formatted_path = []
  for i in range(len(path)):
    obj = path[i]

    if len(obj) == 3:
      len = obj[1]
      alpha_1 = obj[2]
      
      formatted_path.append([
         obj[0] + (len[0] * math.cos(math.radians(alpha_1[1])), alpha_1[0] * math.sin(math.radians(alpha_1[1]))),
         obj[0],
         obj[0] - (len[1] * math.cos(math.radians(alpha_1[1])), alpha_1[0] * math.sin(math.radians(alpha_1[1])))
      ])
    else:
      len = obj[1]
      alpha_1 = obj[2]
      alpha_2 = obj[3]

      formatted_path.append([
         obj[0] + (len[0] * math.cos(math.radians(alpha_1[1])), alpha_1[0] * math.sin(math.radians(alpha_1[1]))),
         obj[0],
         obj[0] + (len[1] * math.cos(math.radians(alpha_2[1])), alpha_2[0] * math.sin(math.radians(alpha_2[1])))
      ])
