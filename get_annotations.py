#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from collections import defaultdict, namedtuple
import json
import logging
import os
import re
import requests
import sys
import urllib.parse
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

CategoryLabel = namedtuple('CategoryLabel', ['color', 'name', 'label'])


LABEL_NAME_TO_RGB = {
    'article': (255, 165, 0),
    'image': (0, 0, 255),
    'advertisement': (0, 128, 0),
    'additional': (128, 0, 128),
}


def get_query_value(url, query_key):
    parse_result = urllib.parse.urlparse(url)
    query_dict = urllib.parse.parse_qs(parse_result.query)
    if query_key in query_dict:
        # unquote_plus decodes the %2B to +, which unquote does not.
        # XXX: Unquoting again should not even be necessary, but the
        # query string seems to have been doubly quoted.
        return urllib.parse.unquote_plus(query_dict[query_key][0])
    return None


def get_image_dimensions(image_path):
    # Width, height
    return Image.open(image_path).size


class Annotation:

    def __init__(self, id, sources, selectors, labels):
        lengths = (len(sources), len(selectors), len(labels))
        if not all(le == lengths[0] for le in lengths[1:]):
            raise ValueError(
                'sources, selectors and labels must have the same length,'
                ' but their lengths are {}'.format(lengths)
            )

        self.sources = sources
        self.selectors = selectors
        self.labels = labels
        self.image_paths = []

        self.get_polygons()

    def find_corresponding_images(self, publication_top_dir):
        self.image_paths = []
        for source in self.sources:
            remote_path = get_query_value(source, 'IIIF')
            match = re.match(
                r'^imageStorage/ecpo_new/[^/]+/(?P<fname>[^(\.tif)]+).tif/.*$',
                remote_path
            )
            name_no_ext = match.group('fname')
            name = name_no_ext + '.jpg'
            local_path = os.path.join(publication_top_dir, name)
            self.image_paths.append(local_path)

    def get_polygons(self):
        self.polygons = []
        for selector in self.selectors:
            polygon = []
            root = ET.fromstring(selector)
            match = re.match(
                r'matrix\((?P<a>[\d.]+) (?P<b>[\d.]+) (?P<c>[\d.]+)'
                r' (?P<d>[\d.]+) (?P<e>[\d.]+) (?P<f>[\d.]+)\)',
                root.attrib.get('transform', '')
            )
            if match:
                # These make up a transformation matrix of the shape
                # a c e
                # b d f
                # A vector (x y) is transformed by this to be:
                # x' = a x + c y + e
                # y' = b x + d y + f
                trans = {}
                trans['a'] = float(match.group('a'))
                trans['b'] = float(match.group('b'))
                trans['c'] = float(match.group('c'))
                trans['d'] = float(match.group('d'))
                trans['e'] = float(match.group('e'))
                trans['f'] = float(match.group('f'))
            else:
                logging.warning(
                    'Selector {} does not specify a transformation')
                trans = None

            polygon_elm = root.find('polygon')
            if polygon_elm is not None:
                for x_y in polygon_elm.attrib['points'].split():
                    x, y = x_y.split(',')
                    x = float(x)
                    y = float(y)

                    if trans:
                        polygon.append(
                            (trans['a'] * x + trans['c'] * y + trans['e'],
                             trans['b'] * x + trans['d'] * y + trans['f'])
                        )
                    else:
                        polygon.append((x, y))
                self.polygons.append(polygon)
            else:
                logging.warning('No polygon found in selector {}'
                                .format(selector))


class AnnotationPage:

    def __init__(self, url):
        self.url = url
        self.content = self.download_page()

    def download_page(self):
        response = requests.get(self.url)
        try:
            content = response.json()
        except json.JSONDecodeError:
            raise RuntimeError('Did not get valid JSON response from {}'
                               .format(self.url))
        return content

    def is_last_page(self):
        return self.content['id'] == self.content['last']

    def next_url(self):
        if self.content and 'next' in self.content:
            next_url_parsed = urllib.parse.urlparse(self.content['next'])
            # Since the URLs in the response point to localhost:8080 on HTTP,
            # we need to change domain, port and scheme.
            original_url_parsed = urllib.parse.urlparse(self.url)
            actual_next_url_parsed = urllib.parse.ParseResult(
                scheme=original_url_parsed.scheme,
                netloc=original_url_parsed.netloc,
                path=next_url_parsed.path,
                params=next_url_parsed.params,
                query=next_url_parsed.query,
                fragment=next_url_parsed.fragment,
            )
            return actual_next_url_parsed.geturl()
        return None

    def get_annotations(self, publication_top_dir):
        for item in self.content['items']:
            if len(item['body']) != len(item['target']):
                logging.warning(
                    'item["body"] and item["target"] have different lengths.'
                    ' Full item: {}'.format(item)
                )
                # This may be not an Annotation object.
                # Just continue with the next one.
                continue

            annotation_id = item['id']
            sources = []
            selectors = []
            labels = []
            for body_elm, target_elm in zip(item['body'], item['target']):
                label = CategoryLabel(
                    color=body_elm['value']['color'],
                    name=body_elm['value']['name'],
                    label=body_elm['value']['label']
                )
                sources.append(target_elm['source'])
                selectors.append(target_elm['selector']['value'])
                labels.append(label)

            annotation = Annotation(
                id=annotation_id, sources=sources, selectors=selectors,
                labels=labels
            )
            annotation.find_corresponding_images(publication_top_dir)
            yield annotation


def get_annotations(publication_top_dir, base_url):
    page = AnnotationPage(base_url)
    yield from page.get_annotations(publication_top_dir)
    while not page.is_last_page():
        page = AnnotationPage(page.next_url())
        yield from page.get_annotations(publication_top_dir)


def construct_mask(reference_image_path, annotations,
                   label_name_to_rgb=LABEL_NAME_TO_RGB,
                   restrict_to_label_names=None):
    width, height = get_image_dimensions(reference_image_path)
    mask = Image.new('RGB', (width, height), color=0)
    draw = ImageDraw.Draw(mask, 'RGB')
    for annotation in annotations:
        for polygon, label in zip(annotation.polygons, annotation.labels):
            if (restrict_to_label_names
                    and label.name not in restrict_to_label_names):
                continue
            draw.polygon(polygon, label_name_to_rgb[label.name])
    return mask


def main(publication_top_dir, max_annotations, restrict_to_label_names,
         base_url, out_top_dir=None):
    max_annotations = 5000
    publication_top_dir = os.path.abspath(publication_top_dir)
    if not out_top_dir:
        out_top_dir = os.path.join(
            # os.path.normpath is necessary to remove a possible trailing
            # slash before calling os.path.dirname
            os.path.dirname(os.path.normpath(publication_top_dir)),
            'masks'
        )

    image_path_to_annotations = defaultdict(list)
    # This loop assumes that there will be exactly one source for each
    # annotation.
    annotations = get_annotations(publication_top_dir, base_url)
    for i, annotation in enumerate(annotations):
        image_path = annotation.image_paths[0]
        logging.info('Found annotation for {}'.format(image_path))
        image_path_to_annotations[image_path].append(annotation)
        if i == max_annotations:
            break

    for image_path, annotations in image_path_to_annotations.items():
        mask = construct_mask(image_path, annotations,
                              restrict_to_label_names=restrict_to_label_names)
        mask_path_wrong_ext = os.path.join(
            out_top_dir,
            os.path.relpath(image_path, publication_top_dir)
        )
        mask_path_base, _ = os.path.splitext(mask_path_wrong_ext)
        mask_path = mask_path_base + '.png'
        logging.info('Saving mask to {}'.format(mask_path))
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        mask.save(mask_path, 'PNG')


def parse_list(list_as_str, sep=','):
    return [part.strip() for part in list_as_str.split(sep)]


def parse_args():
    description = ('Download annotations from the ECPO API and store them as '
                   'mask PNG images.')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('publication_top_dir',
                        help='Local directory where all images are stored.'
                        ' XXX: Currently only the top level directory for one'
                        ' newspaper is supported. This needs to be changed.')
    parser.add_argument('--max-annotations', '-m', type=int, default=10000,
                        help='Maximum single annotations to extract. The'
                        ' program will exit after that many annotations have'
                        ' been extracted even if that means that some images'
                        ' are missing some annotations.')
    parser.add_argument('--out-top-dir',
                        help='Where to save the mask images. If not given, it'
                        ' will save them under the directory masks next to the'
                        ' publication_top_dir')
    parser.add_argument('--restrict-to-label-names', '-l',
                        help='Comma-separated list of label names to extract.'
                        ' All other labels will be ignored when constructing'
                        ' the masks. If not given, all labels will be'
                        ' extracted.')
    parser.add_argument('--base-url',
                        default='https://ecpo.existsolutions.com/exist/apps/wap/annotations/',
                        help='URL of the Annotations API endpoints')
    args = parser.parse_args()
    args.restrict_to_label_names = parse_list(args.restrict_to_label_names)
    return args


if __name__ == '__main__':
    ARGS = parse_args()
    main(**vars(ARGS))
