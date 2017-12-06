import os
import numpy as np

COLORS = [(255, 0, 0), (102, 51, 51), (229, 130, 115), (178, 71, 0), (102, 41, 0), (255, 166, 64), (115, 96, 57),
              (202, 217, 0), (95, 102, 0), (238, 242, 182), (113, 179, 89), (77, 102, 77), (0, 255, 34), (0, 255, 170),
              (35, 140, 105), (191, 255, 234), (0, 190, 204), (0, 77, 115), (61, 182, 242), (182, 214, 242),
              (77, 90, 102), (0, 31, 115), (102, 129, 204), (92, 51, 204), (204, 0, 255), (218, 121, 242),
              (217, 163, 213), (102, 77, 100), (128, 0, 102), (255, 0, 170), (242, 0, 97), (242, 121, 170),
              (166, 0, 22)]

COLORS_INDEX = {}
for i in range(len(COLORS)):
    COLORS_INDEX[COLORS[i]] = i

LABELS = ['awning', 'balcony', 'bird', 'boat', 'bridge', 'building', 'bus', 'car', 'cow', 'crosswalk', 'desert', 'door', 'fence', 'field', 'grass', 'moon', 'mountain', 'person', 'plant', 'pole', 'river', 'road', 'rock', 'sand', 'sea', 'sidewalk', 'sign', 'sky', 'staircase', 'streetlight', 'sun', 'tree', 'window']

IMG_DIR = os.path.join(os.getcwd(), 'data', 'SiftFlowDataSet', 'Images',
                           'spatial_envelope_256x256_static_8outdoorcategories')

SEG_DIR = os.path.join(os.getcwd(), 'data', 'segmented')

MEAN = np.array((104.00698793, 116.66876762, 122.67891434))
