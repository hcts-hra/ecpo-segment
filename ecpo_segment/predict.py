#/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from glob import glob

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

from dh_segment.inference import LoadedModel
from dh_segment.post_processing import boxes_detection, binarization
from dh_segment.utils.labels import get_classes_color_from_file


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div


def main(input_dir, output_dir, model_dir, classes_file):
    if not os.path.isdir(input_dir):
        print('No such input directory: {}'.format(input_dir))
        sys.exit(1)

    if not os.path.isdir(model_dir):
        print('No such model directory: {}'.format(model_dir))
        sys.exit(2)

    if not os.path.isfile(classes_file):
        print('No such classes file: {}'.format(classes_file))
        sys.exit(3)

    input_files = glob('{}/*'.format(input_dir))

    raw_dir = os.path.join(output_dir, 'raw')
    raw_overlays_dir = os.path.join(output_dir, 'raw_overlays')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(raw_overlays_dir, exist_ok=True)

    # Shape [num_classes, 3] (3 is for RGB)
    class_colors = np.array(
        get_classes_color_from_file(classes_file),
        dtype=np.uint8
    )
    num_classes = class_colors.shape[0]

    with tf.Session():
        m = LoadedModel(model_dir, predict_mode='filename')

        for filename in tqdm(input_files, desc='Processed files'):
            rootname, _ = os.path.splitext(filename)
            basename = os.path.basename(rootname + '.png')

            # For each image, predict each pixel's label
            prediction_outputs = m.predict(filename)
            probs = prediction_outputs['probs'][0]
            original_shape = prediction_outputs['original_shape']
            assert probs.shape[2] == num_classes

            # Shape: (h, w)
            class_map = probs.argmax(axis=-1)

            # Upscale to have full resolution image (cv2 uses (w,h) and not (h,w) for giving shapes)
            class_map_upscaled = cv2.resize(
                class_map.astype(np.uint8, copy=False),
                tuple(original_shape[::-1]),
                interpolation=cv2.INTER_NEAREST
            )
            # Shape: (h', w', 3)
            color_map = np.take(class_colors, class_map_upscaled, axis=0)
            raw = Image.fromarray(color_map)
            raw.save(os.path.join(raw_dir, basename), 'PNG')
            Image.fromarray(color_map)

            original_img = Image.open(filename).convert('RGBA')
            predicted_mask = Image.fromarray(color_map).convert('RGBA')
            raw_overlay = Image.blend(original_img, predicted_mask, 0.5)
            raw_overlay.save(os.path.join(raw_overlays_dir, basename), 'PNG')


def parse_args():
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('input_dir', help='Directory containing input images')
    parser.add_argument('output_dir',
                        help='Directory which will contain the outputs')
    parser.add_argument('--model-dir', help='Directory containing the model',
                        default='model')
    parser.add_argument('--classes-file', help='The classes.txt file',
                        default='classes.txt')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    ARGS = parse_args()
    main(**vars(ARGS))
