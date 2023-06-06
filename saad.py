import numpy as np
sent = "<BOS> <OBJ> <FLAT> 58.91 56.41 4.26 3.79 (44.2,) <FLAT> 50.18 50.8 1.08 4.71 (98.56,) <FLAT> 10.06 57.64 1.51 3.1 (145.12,) <FLAT> 23.67 21.23 0.25 1.45 (212.47,) <FLAT> 87.09 41.12 4.59 3.56 (216.39,) <FLAT> 58.91 56.41 0.63 3.05 (178.2,) <FLAT> 50.18 50.8 0.12 0.14 (236.15,) <FLAT> 10.06 57.64 2.28 0.15 (51.71,) <FLAT> 23.67 21.23 2.41 0.09 (66.43,) <FLAT> 87.09 41.12 0.19 3.88 (260.87,) <FLAT> 58.91 56.41 4.81 1.9 (113.3,) <FLAT> 50.18 50.8 1.57 2.7 (24.7,) <FLAT> 10.06 57.64 1.66 2.49 (127.65,) <FLAT> 23.67 21.23 0.49 4.87 (83.76,) <FLAT> 87.09 41.12 0.63 3.65 (194.51,) <FLAT> 58.91 56.41 4.19 2.27 (232.24,) <FLAT> 50.18 50.8 3.57 2.22 (287.07,) <FLAT> 10.06 57.64 3.43 3.41 (58.94,) <FLAT> 23.67 21.23 3.88 3.57 (316.12,) <FLAT> 87.09 41.12 4.23 3.85 (231.89,) <FLAT> 58.91 56.41 0.47 3.23 (115.75,) <FLAT> 50.18 50.8 2.4 0.33 (123.41,) <FLAT> 10.06 57.64 1.84 3.18 (190.72,) <FLAT> 23.67 21.23 4.65 0.09 (142.71,) <FLAT> 87.09 41.12 4.65 0.32 (111.52,) <FLAT> 58.91 56.41 0.04 2.94 (134.64,) <FLAT> 50.18 50.8 4.64 3.03 (113.29,) <FLAT> 10.06 57.64 3.17 1.63 (16.33,) <FLAT> 23.67 21.23 2.59 1.0 (54.82,) <FLAT> 87.09 41.12 0.79 0.34 (251.14,) <FLAT> 58.91 56.41 1.33 1.69 (158.5,) <FLAT> 50.18 50.8 2.5 0.37 (173.74,) <FLAT> 10.06 57.64 4.03 1.08 (333.74,) <SHARP> 23.67 21.23 3.95 4.17 (165.74,) (197.79,) <FLAT> 87.09 41.12 2.89 4.19 (176.52,) <FLAT> 58.91 56.41 4.96 4.37 (317.89,) <FLAT> 50.18 50.8 4.67 3.12 (344.49,) <FLAT> 10.06 57.64 2.42 0.2 (297.69,) <FLAT> 23.67 21.23 4.9 2.18 (191.92,) <FLAT> 87.09 41.12 3.95 2.0 (209.52,) <FLAT> 58.91 56.41 0.46 4.25 (39.23,) <FLAT> 50.18 50.8 4.27 0.57 (103.64,) <FLAT> 10.06 57.64 0.57 0.84 (252.1,) <FLAT> 23.67 21.23 0.2 2.28 (130.46,) <FLAT> 87.09 41.12 1.45 1.97 (28.28,) <FLAT> 58.91 56.41 4.55 2.56 (189.75,) <FLAT> 50.18 50.8 2.49 2.1 (178.94,) <FLAT> 10.06 57.64 1.15 0.53 (222.71,) <SHARP> 23.67 21.23 4.05 2.94 (227.93,) (298.49,) <FLAT> 87.09 41.12 2.87 2.55 (99.87,)  <CLR> 0 0 0 <EOS>"


def parse_single_object(obj):
    obj = obj.replace("(", "").replace(",)", "")
    obj = obj.split(' ')
    obj = np.array(obj)
    obj = np.split(obj, np.where(obj == '<CLR>')[0])
    obj = [obj[obj != '<CLR>'] for obj in obj]
    return [obj[0], obj[1]]


def parse_sentence(sentence):
    # Remove the <BOS> and <EOS> tokens
    sentence = sentence.replace("(", "").replace(",)", "")
    sentence = sentence.replace("  <CLR>", " <CLR>")

    #  split by <OBJ> token
    objects = sentence.split(' <OBJ> ')[1:]

    # Parse each object
    objects = [parse_single_object(obj) for obj in objects]

    objects_ = [obj[0] for obj in objects]
    colors_ = [obj[1] for obj in objects]
   
    # Filter <EOS> token from colors array
    colors_ = [color[color != '<EOS>'] for color in colors_]

    # Split objects_ array by <FLAT> and <SHARP> tokens
    objects_ = np.array(objects_)
    objects_ = [np.split(obj, 
      # union of <FLAT> and <SHARP> tokens
      np.union1d(
         np.where(obj == '<FLAT>')[0],
         np.where(obj == '<SHARP>')[0]
      )) for obj in objects_]
    objects_ = [obj[1:] for obj in objects_]
    objects_ = [[obj[obj != '<FLAT>'] for obj in obj] for obj in objects_]
    objects_ = [[obj[obj != '<SHARP>'] for obj in obj] for obj in objects_]
    
    objects_ = [[list(np.array(obj).astype('float32')) for obj in obj] for obj in objects_]

    return objects_, [np.array(obj).astype('float32')for obj in colors_]


print(parse_sentence(sent))
