import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pix2vectorseq.draw import drawShapeFromPoints, drawPath, format_path, transform_matrix
from pix2vectorseq.bezier import Bezier
from pix2vectorseq.utils import generate_random_concave_polygon, randomSegmentPoints

def test():
   # Example usage
   points = randomSegmentPoints((0, 100, 0, 100), 6)
   path_ = drawShapeFromPoints(points)
   formatted_path = format_path(path_)
   path = transform_matrix(formatted_path)

   drawPath(path)

if __name__ == '__main__':
    test()
