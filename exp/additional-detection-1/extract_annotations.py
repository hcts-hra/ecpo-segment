#!/usr/bin/env python

import argparse
import os
from glob import glob

import cv2
from imageio import imread, imsave
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

from dh_segment.inference import LoadedModel
from dh_segment.post_processing import boxes_detection, binarization


def make_binary_mask(probs: np.ndarray, threshold: float=-1) -> np.ndarray:
    """
    Computes the binary mask of the detected class from probabilities
    :param probs: array with values in range [0, 1]
    :param threshold: threshold between [0 and 1], if negative Otsu's adaptive threshold will be used
    :return: binary mask
    """
    mask = binarization.thresholding(probs, threshold)
    mask = binarization.cleaning_binary(mask, kernel_size=5)
    return mask


def main(input_dir, model_dir, out_dir, raw_out_dir=None, min_area=0.0005,
         overlay_alpha=127, box_color=(255, 0, 0)):
    os.makedirs(out_dir, exist_ok=True)
    if raw_out_dir:
        os.makedirs(raw_out_dir, exist_ok=True)
    input_files = glob('{}/*'.format(input_dir))
    with tf.Session():
        # Load the model
        m = LoadedModel(model_dir, predict_mode='filename')
        for filename in tqdm(input_files, desc='Processed files'):
            basename = os.path.basename(filename).split('.')[0]

            # For each image, predict each pixel's label
            prediction_outputs = m.predict(filename)
            probs = prediction_outputs['probs'][0]
            original_shape = prediction_outputs['original_shape']
            # Take only class '1'
            # (class 0 is the background, class 1 is the annotation.)
            probs = probs[:, :, 1]
            probs = probs / np.max(probs)  # Normalize to be in [0, 1]

            # Binarize the predictions
            preds_bin = make_binary_mask(probs)

            # Upscale to have full resolution image
            # (cv2 uses (w,h) and not (h,w) for giving shapes)
            bin_upscaled = cv2.resize(
                preds_bin.astype(np.uint8, copy=False),
                tuple(original_shape[::-1]),
                interpolation=cv2.INTER_NEAREST
            )

            if raw_out_dir:
                # If requested, draw the binary mask as an overlay
                # over the image and save it.
                img = Image.open(filename)
                img = img.convert('RGBA')
                overlay_arr = np.stack(
                    [
                        bin_upscaled * box_color[0],  # R
                        bin_upscaled * box_color[1],  # G
                        bin_upscaled * box_color[2],  # B
                        np.ones_like(bin_upscaled) * overlay_alpha  # A
                    ],
                    axis=2
                )
                overlay = Image.fromarray(overlay_arr, mode='RGBA')
                img.paste(overlay, (0, 0), overlay)
                img.save(
                    os.path.join(raw_out_dir, '{}_raw.png'.format(basename)),
                    'PNG'
                )

            # Find quadrilateral enclosing the page
            boxes = boxes_detection.find_boxes(
                bin_upscaled.astype(np.uint8, copy=False),
                min_area=min_area,
                mode='min_rectangle',
            )

            # Draw boxes on original image.
            original_img = imread(filename, pilmode='RGB')
            if boxes is not None:
                cv2.polylines(
                    original_img,
                    boxes,
                    True,
                    box_color,
                    thickness=5
                )
            else:
                print('No annotation found in {}'.format(filename))

            imsave(os.path.join(out_dir, '{}_boxes.jpg'.format(basename)),
                   original_img)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Use trained model to annotate images with a single class')
    parser.add_argument('input_dir',
                        help='Directory containing the input images')
    parser.add_argument('-m', '--model-dir', required=True,
                        help='Directory containing the trained model')
    parser.add_argument('-o', '--out-dir', default='processed_images',
                        help='Directory to store annotated images in')
    parser.add_argument('--raw-out-dir',
                        help='Directory to store annotated images in')
    parser.add_argument(
        '-a', '--min-area', default=0.0005, type=float,
        help='Minimum area fraction of the image an annotation has to cover.'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    ARGS = parse_args()
    main(**vars(ARGS))
