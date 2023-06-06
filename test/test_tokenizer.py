import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pix2vectorseq.bezier import Bezier
from pix2vectorseq.dataset import CFG, VectorTokenizer, parse_sentence
from pix2vectorseq.draw import (drawPath, drawShapeFromPoints, format_path,
                                transform_matrix)
from pix2vectorseq.utils import (generate_random_concave_polygon,
                                 generateSentence, randomSegmentPoints, chunk_array)
import torch

import numpy as np

def test():
   # Example usage
   points = randomSegmentPoints((0, 100, 0, 100), 6)
   path_ = drawShapeFromPoints(points)

   formatted_path_ = format_path(path_)
   path__ = transform_matrix(formatted_path_)
   drawPath(path__)

   sentence = generateSentence(path_)

   print("Generated sentence: ", sentence)

   tokenizer = VectorTokenizer(num_bins=CFG.num_bins,
                          width=CFG.img_size, height=CFG.img_size, max_len=CFG.max_len)
   CFG.pad_idx = tokenizer.PAD_code

   objects, colors = parse_sentence(sentence)

   print("Parsed objects: ", objects)

   tokenized = torch.Tensor(tokenizer(objects, [[0, 0, 0]]))

   print("Tokenized objects: ", tokenized)

   decoded_objects, decoded_colors = tokenizer.decode(tokenized)

   print("Decoded objects: ", decoded_objects)

   # replace all tokenizer.FLAT_code with 999999 using numpy
   decoded_objects = np.where(decoded_objects == tokenizer.FLAT_code, 999999, decoded_objects)

   print("Decoded objects (processed): ", decoded_objects)

   chunked_objects = chunk_array(decoded_objects[0])

   print("Chunked objects: ", chunked_objects)

   formatted_path = format_path(chunked_objects)

   print("Formatted path: ", formatted_path)

   path = transform_matrix(formatted_path)

   drawPath(path)

if __name__ == '__main__':
    test()