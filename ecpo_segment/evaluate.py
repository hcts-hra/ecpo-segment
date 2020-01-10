#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import sys
from typing import Dict, Tuple

from PIL import Image
from dh_segment.utils.labels import get_classes_color_from_file
import numpy as np


def read_masks(predicted_masks_dir, reference_masks_dir):
    predicted_masks_files = {
        os.path.basename(path): path
        for path
        in glob.glob('{}/*png'.format(os.path.normpath(predicted_masks_dir)))
    }
    reference_masks_files = {
        os.path.basename(path): path
        for path
        in glob.glob('{}/*png'.format(os.path.normpath(reference_masks_dir)))
    }

    if set(predicted_masks_files.keys()) == set(reference_masks_files.keys()):
        masks = []
        for basename in predicted_masks_files:
            predicted_mask = Image.open(predicted_masks_files[basename])
            reference_mask = Image.open(reference_masks_files[basename])
            masks.append((np.array(predicted_mask), np.array(reference_mask)))
    else:
        print('Set of files in {} and {} is not identical.'
              .format(predicted_masks_dir, reference_masks_dir),
              file=sys.stderr)
        sys.exit(1)

    return masks


def color_mask_to_class_mask(color_mask: np.ndarray,
                             color_dict: Dict[Tuple[int, int, int], int]):
    # color_mask: [W, H, C]
    class_mask = np.zeros(color_mask.shape[:2])  # [W, H]
    for row in range(class_mask.shape[0]):
        for col in range(class_mask.shape[1]):
            wanted_color = tuple(color_mask[row, col])
            if wanted_color in color_dict:
                class_mask[row, col] = color_dict[wanted_color]
            else:
                print('Unexpected color: {}'.format(wanted_color),
                      file=sys.stderr)
                sys.exit(1)
    return class_mask


def get_class_masks(color_masks, color_dict):
    class_masks = []
    for predicted_color_mask, reference_color_mask in color_masks:
        predicted_class_mask = color_mask_to_class_mask(predicted_color_mask,
                                                        color_dict)
        reference_class_mask = color_mask_to_class_mask(reference_color_mask,
                                                        color_dict)
        class_masks.append((predicted_class_mask, reference_class_mask))

    return class_masks


def iou_for_class(pred_mask, ref_mask, class_):
    pred_mask = pred_mask == class_
    ref_mask = ref_mask == class_
    intersection = pred_mask & ref_mask
    union = pred_mask | ref_mask
    return intersection.sum() / union.sum()


def main(predicted_masks_dir, reference_masks_dir, classes_file):
    print('Reading color masks')
    color_masks = read_masks(predicted_masks_dir, reference_masks_dir)
    class_colors = get_classes_color_from_file(classes_file)
    color_dict = {tuple(color): class_
                  for class_, color in enumerate(class_colors)}
    print('Converting color masks to class masks')
    class_masks = get_class_masks(color_masks, color_dict)

    print('Computing IoU per class')
    ious_per_class = {class_: [] for _, class_ in color_dict.items()}
    for pred_mask, ref_mask in class_masks:
        for class_, ious in ious_per_class.items():
            ious.append(iou_for_class(pred_mask, ref_mask, class_))

    mean_iou_per_class = {class_: np.mean(ious)
                          for class_, ious in ious_per_class.items()}
    for class_, iou in mean_iou_per_class.items():
        print('Mean IoU for class {}: {:.3f}'.format(class_, iou))


def parse_args():
    description = 'Evaluate a document segmentation against the reference.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('predicted_masks_dir')
    parser.add_argument('reference_masks_dir')
    parser.add_argument('classes_file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    ARGS = parse_args()
    main(**vars(ARGS))
