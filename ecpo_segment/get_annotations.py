#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# XXX, FIXME
# This program has a couple of issues that should be addressed in further
# development of the project:
#
# 1. One-periodical-only assumption
#
# How it is: Especially in the function `find_corresponding_images`, it is
# assumed that all annotations in the API correspond to the same periodical
# (e.g. Jingbao) and the user is assumed to give the path to the corresponding
# local directory in his commandline arguments.
#
# How it should be: The user should only point to the top-level ECPO directory
# containing images of all periodicals and the correct image should be found
# from the information given in the JSON annotation object.
#
# 2. Downloading of all annotations across folds
#
# How it is: The program just downloads all annotations it can find on the
# /annotations/ endpoint. This is problematic (1) because the user cannot make
# more fine-grained queries and (2) because it means holding a lot of
# annotations in RAM at the same time until all annotations for a given fold
# are found, eventually leading to an out-of-memory situation for very large
# amounts of available annotations.
#
# How it should be: The user should be able to specify for which periodical
# they want to retrieve annotations (cf. problem 1). Even more importantly, the
# program should use fold-specific endpoints to download all annotations for
# one fold at a time and then saving them to disk instead of downloading
# everything into memory and saving everything to disk in the end.
#
# 3. Assumption of local presence of images
#
# How it is: It is assumed that the user has a local copy of the images for all
# folds for which they want to retrieve annotations.
#
# How it should be: The user should be able to specify that they want to
# download the source images along with their annotations. This shouldn’t be
# the default because it will considerably increase download time and load on
# the server.

import argparse
from collections import defaultdict, namedtuple
import json
import logging
import os
import re
import shutil
import sys
from typing import (Collection, Generator, Iterable, List, Mapping, Optional,
                    Sequence, Tuple)
import urllib.parse
import xml.etree.ElementTree as ET

import requests
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

CategoryLabel = namedtuple('CategoryLabel', ['color', 'name', 'label'])


# Mapping from a label name to the RGB color the annotation is going
# to have in the mask.
LABEL_NAME_TO_RGB = {
    'article': (255, 165, 0),
    'image': (0, 0, 255),
    'advertisement': (0, 128, 0),
    'additional': (128, 0, 128),
}


def get_query_value(url: str, query_key: str,
                    doubly_encoded: bool = True) -> Optional[str]:
    """Get a value from a URL’s doubly-encoded query string using the key.

    :param url: A URL
    :param query_key: A key / variable name in the query string
    :param doubly_encoded: Whether the query string of the URL is
        doubly URL-encoded. If this is True, the value is URL-decoded
        before return.
    :return: The retrieved value or None
    """
    parse_result = urllib.parse.urlparse(url)
    query_dict = urllib.parse.parse_qs(parse_result.query)
    if query_key in query_dict:
        val = query_dict[query_key][0]
        if doubly_encoded:
            # unquote_plus decodes the %2B to +, which unquote does not.
            val = urllib.parse.unquote_plus(val)
        return val
    return None


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Read an image from disk and get its width and height.

    :param image_path: Path to an image
    :return: The image’s width and height as a tuple.
    """
    return Image.open(image_path).size


class Annotation:
    """Our Python counterpart of the ECPO API’s Annotation item.

    :param id: The annotation item’s ID
    :param sources: The annotation item’s targets’ sources
    :param selectors: The annotation item’s targets’ selectors
    :param labels: The CategoryLabel objects in the annotation item’s bodies.
    """

    def __init__(self, id: str, sources: Sequence[str],
                 selectors: Sequence[str], labels: Sequence[CategoryLabel]):
        """Initialize the Annotation object."""
        lengths = (len(sources), len(selectors), len(labels))
        if not all(le == lengths[0] for le in lengths[1:]):
            raise ValueError(
                'sources, selectors and labels must have the same length,'
                ' but their lengths are {}'.format(lengths)
            )

        self.id = id
        self.sources = sources
        self.selectors = selectors
        self.labels = labels
        self.image_paths = []

        self.get_polygons()

    def find_corresponding_images(self, publication_top_dir: str) -> None:
        """Find images corresponding to sources in publication_top_dir.

        The method finds the corresponding images by comparing the filenames of
        the local files with the source URLs. Instead of returning something,
        this method modifies the `self.image_paths` attribute

        :param publication_top_dir: Local top level directory where the images
            of the publications for this document type are stored. E.g.
            ~/ECPO/Jingbao

        """
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
        """Get polygons from the annotation.

        The polygons are constructed from the selector and stored in the
        `self.polygons` attribute.
        """
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
    """A page listing many annotations and containing a link to the next page.

    A sensible value for initializing this could be
    https://ecpo.existsolutions.com/exist/apps/wap/annotations/.
    """

    def __init__(self, url: str) -> None:
        """Initialize the AnnotationPage and download its content.

        :param url: This annotation listing page’s URL
        """
        self.url = url
        self.content = self.download_page()

    def download_page(self) -> dict:
        """Download the JSON annotation listing from the internet.

        :return: The content parsed into a dict
        """
        response = requests.get(self.url)
        try:
            content = response.json()
        except json.JSONDecodeError:
            raise RuntimeError('Did not get valid JSON response from {}'
                               .format(self.url))
        return content

    def is_last_page(self) -> bool:
        """Find out if this page is the last annotation page."""
        return self.content['id'] == self.content['last']

    def next_url(self) -> Optional[str]:
        """Get the URL of the next page of annotations.

        :return: The URL or None if this is already the last page.
        """
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

    def get_annotations(self, publication_top_dir: str) -> Generator[Annotation, None, None]:
        """Get annotations from this page.

        :param publication_top_dir: Local top level directory where the images
            of the publications for this document type are stored. E.g.
            ~/ECPO/Jingbao
        :yield: The annotations, one at a time
        """
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


def get_annotations(publication_top_dir: str,
                    base_url: str) -> Generator[Annotation, None, None]:
    """Download annotations from the ECPO API.

    :param publication_top_dir: Local top level directory where the images of
        the publications for this document type are stored. E.g. ~/ECPO/Jingbao

    :param base_url: API URL for listing the annotations
    :yield: The annotations, one at a time
    """
    page = AnnotationPage(base_url)
    yield from page.get_annotations(publication_top_dir)
    while not page.is_last_page():
        page = AnnotationPage(page.next_url())
        yield from page.get_annotations(publication_top_dir)


def construct_mask(width: int, height: int, annotations: Iterable[Annotation],
                   label_name_to_rgb: Mapping[str, Tuple[int, int, int]] = LABEL_NAME_TO_RGB,
                   restrict_to_label_names: Optional[Collection[str]] = None) -> Image:
    """Construct a mask image from annotations.

    :param width: Width of the resulting image
    :param height: Height of the resulting image
    :param annotations: Annotation objects
    :param label_name_to_rgb: Mapping from label name to RGB tuple
    :param restrict_to_label_names: Names of the labels that are rendered in
        the mask. If None (the default), all labels are rendered.
    :return: Constructed mask
    """
    mask = Image.new('RGB', (width, height), color=0)
    draw = ImageDraw.Draw(mask, 'RGB')
    for annotation in annotations:
        for polygon, label in zip(annotation.polygons, annotation.labels):
            if (restrict_to_label_names
                    and label.name not in restrict_to_label_names):
                continue
            draw.polygon(polygon, label_name_to_rgb[label.name])
    return mask


def make_dir_structure(paths, link_dir=None, copy_dir=None, flatten=True,
                       ref_src_dir=None):
    if not (link_dir or copy_dir):
        raise ValueError('Either link_dir or copy_dir must be specified')

    if not flatten:
        if not ref_src_dir:
            raise ValueError(
                'If flatten is False, you must specify a ref_src_dir to use as'
                ' the hierarchy’s top dir'
            )

    for path in paths:
        src = os.path.abspath(path)
        if copy_dir:
            os.makedirs(copy_dir, exist_ok=True)
            if flatten:
                dst = os.path.join(copy_dir,
                                    os.path.basename(path))
            else:
                dst = os.path.join(
                    copy_dir,
                    os.path.relpath(path, ref_src_dir)
                )
                os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)

        if link_dir:
            os.makedirs(link_dir, exist_ok=True)
            if flatten:
                dst = os.path.join(link_dir,
                                    os.path.basename(path))
            else:
                dst = os.path.join(
                    link_dir,
                    os.path.relpath(path, ref_src_dir)
                )
                os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst)


def main(publication_top_dir, max_annotations, restrict_to_label_names,
         base_url, nested_dirs=False, mask_dir=None, copy_image_dir=None,
         link_image_dir=None):
    """Download annotations from the ECPO API and save them as mask images."""
    publication_top_dir = os.path.abspath(publication_top_dir)
    if not mask_dir:
        mask_dir = os.path.join(
            # os.path.normpath is necessary to remove a potential trailing slash
            # before calling os.path.dirname, because in Python
            # dirname('/usr/') is '/usr'
            os.path.dirname(os.path.normpath(publication_top_dir)),
            'masks'
        )

    # Download the annotations.
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

    # Construct annotation masks and save them as PNG images.
    for image_path, annotations in image_path_to_annotations.items():
        width, height = get_image_dimensions(image_path)
        mask = construct_mask(width, height, annotations,
                              restrict_to_label_names=restrict_to_label_names)
        if nested_dirs:
            mask_path_wrong_ext = os.path.join(
                mask_dir,
                os.path.relpath(image_path, publication_top_dir)
            )
        else:
            mask_path_wrong_ext = os.path.join(mask_dir,
                                               os.path.basename(image_path))
        mask_path_base, _ = os.path.splitext(mask_path_wrong_ext)
        mask_path = mask_path_base + '.png'
        logging.info('Saving mask to {}'.format(mask_path))
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        mask.save(mask_path, 'PNG')

    make_dir_structure(
        paths=image_path_to_annotations.keys(),
        link_dir=link_image_dir,
        copy_dir=copy_image_dir,
        flatten=not nested_dirs,
        ref_src_dir=publication_top_dir,
    )


def parse_list(list_as_str: str, sep: str = ',') -> List[str]:
    """Parse sep-separated string of components into a list.

    Splits the list at a separator string (, (comma) by default) and returns
    the resulting parts after stripping whitespace.
    """
    return [part.strip() for part in list_as_str.split(sep)]


def parse_args():
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('publication_top_dir',
                        help='Local directory where all images are stored.'
                        ' XXX: Currently only the top level directory for one'
                        ' newspaper is supported. This needs to be changed.')
    parser.add_argument('--max-annotations', '-m', type=int, default=10000,
                        help='Maximum single annotations to extract. The'
                        ' program will exit after that many annotations have'
                        ' been extracted even if that means that some images'
                        ' are missing some annotations.')
    parser.add_argument('--mask-dir',
                        help='Where to save the mask images. If not given, it'
                        ' will save them under the directory masks next to the'
                        ' publication_top_dir')
    parser.add_argument('--restrict-to-label-names', '-l',
                        help='Comma-separated list of label names to extract.'
                        ' All other labels will be ignored when constructing'
                        ' the masks. If not given, all labels will be'
                        ' extracted.')
    parser.add_argument('--nested-dirs', default=False, action='store_true',
                        help='The mask output directory and the copy_image_dir'
                        ' and link_image_dir will be nested like the'
                        ' publication_top_dir if this flag is set. If it is'
                        ' not set, these directories will be flat.')
    parser.add_argument('--link-image-dir', help='Link the images for which'
                        ' annotations were found into this this directory.')
    parser.add_argument('--copy-image-dir', help='Copy the images for which'
                        ' annotations were found into this this directory.')
    parser.add_argument('--base-url',
                        default='https://ecpo.existsolutions.com/exist/apps/wap/annotations/',
                        help='URL of the Annotations API endpoints')
    args = parser.parse_args()
    if args.restrict_to_label_names:
        args.restrict_to_label_names = parse_list(args.restrict_to_label_names)
    return args


if __name__ == '__main__':
    ARGS = parse_args()
    main(**vars(ARGS))
