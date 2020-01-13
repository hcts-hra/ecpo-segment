#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

import numpy as np
import pytest
from PIL import Image
import responses

from ecpo_segment.get_annotations import (
    Annotation, AnnotationPage, CategoryLabel, construct_mask, get_query_value,
    make_dir_structure, parse_list
)


ANNOTATIONS_URL = 'https://example.com/exist/apps/wap/annotations/'

with open(os.path.join(os.path.dirname(__file__),
                       'annotation_page.json')) as f:
    ANNOTATION_PAGE_CONTENT = json.load(f)

FULL_ANNOTATION_MASK = Image.open(
    os.path.join(os.path.dirname(__file__), 'full_mask.png')
)

ARTICLE_ANNOTATION_MASK = Image.open(
    os.path.join(os.path.dirname(__file__), 'article_mask.png')
)

@pytest.fixture
@responses.activate
def annotation_page():
    responses.add(
        responses.GET,
        ANNOTATIONS_URL,
        json=ANNOTATION_PAGE_CONTENT,
        status=200
    )
    return AnnotationPage(ANNOTATIONS_URL)


@pytest.fixture
def annotation():
    ann = ANNOTATION_PAGE_CONTENT['items'][0]
    target = ann['target'][0]
    label_dict = ann['body'][0]['value']
    label = CategoryLabel(**label_dict)
    return Annotation(
        id=ann['id'],
        sources=[target['source']],
        selectors=[target['selector']['value']],
        labels=[label]
    )


def test_download_annotations_page(annotation_page):
    assert annotation_page.url == ANNOTATIONS_URL
    annotation_page.content == ANNOTATION_PAGE_CONTENT


def test_next_url(annotation_page):
    assert (annotation_page.next_url()
            ==  'https://example.com/exist/apps/wap/annotations/?page=1')


def test_is_last_page(annotation_page):
    assert annotation_page.is_last_page() is False


def test_get_annotations(annotation_page):
    annotations = list(annotation_page.get_annotations('/tmp/'))
    assert len(annotations) == 9
    assert all(isinstance(a, Annotation) for a in annotations)


def test_get_polygons(annotation):
    assert annotation.id == ANNOTATION_PAGE_CONTENT['items'][0]['id']
    assert len(annotation.polygons) == 1
    polygon = [(1953.7, 2041.8), (2058.8, 2041.8),
               (2058.8, 1824.0), (1953.7, 1824.0)]
    assert len(annotation.polygons[0]) == len(polygon)
    assert all(pytest.approx(annotation.polygons[0][i], polygon[i], 0.1)
               for i in range(len(polygon)))


def test_construct_mask():
    width = 100
    height = 100
    annotation = Annotation('dummy', [], [], [])
    annotation.labels = [
        CategoryLabel('violet', 'additional', 'Additional'),
        CategoryLabel('orange', 'article', 'Article'),
        CategoryLabel('orange', 'article', 'Article'),
        CategoryLabel('blue', 'image', 'Image'),
    ]
    annotation.polygons = [
        [(10, 10), (20, 10), (20, 20), (10, 20)],  # clockwise
        [(50, 50), (50, 99), (99, 99), (99, 50)],  # counterclockwise
        [(50, 10), (99, 10), (99, 20), (50, 20)],  # clockwise
        [(0, 0), (0, 2), (2, 2), (2, 0)],  # counterclockwise
    ]
    full_mask = construct_mask(width, height, [annotation])
    assert (np.array(full_mask) == np.array(FULL_ANNOTATION_MASK)).all()
    article_mask = construct_mask(width, height, [annotation],
                               restrict_to_label_names=['article'])
    assert (np.array(article_mask) == np.array(ARTICLE_ANNOTATION_MASK)).all()


def test_get_query_value():
    innocent_url = 'http://example.com/?foo=bar'
    get_query_value(innocent_url, 'foo') == 'bar'
    get_query_value(innocent_url, 'bar') is None

    expected = 'jb_3804_1939-04-30_0005+0008.tif'
    singly_encoded_url = (
        'https://example.com/?IIIF=jb_3804_1939-04-30_0005%2B0008.tif')
    singly_falsely_doubly_decoded =  'jb_3804_1939-04-30_0005 0008.tif'
    assert get_query_value(singly_encoded_url, 'IIIF', False) == expected
    assert (get_query_value(singly_encoded_url, 'IIIF', True)
            == singly_falsely_doubly_decoded)
    assert (get_query_value(singly_encoded_url, 'IIIF')
            == singly_falsely_doubly_decoded)

    doubly_encoded_url = (
        'https://example.com/?IIIF=jb_3804_1939-04-30_0005%252B0008.tif')
    doubly_falsely_singly_decoded = 'jb_3804_1939-04-30_0005%2B0008.tif'
    assert get_query_value(doubly_encoded_url, 'IIIF') == expected
    assert get_query_value(doubly_encoded_url, 'IIIF', True) == expected
    assert (get_query_value(doubly_encoded_url, 'IIIF', False)
            == doubly_falsely_singly_decoded)

    non_existent = 'foobar'
    get_query_value(non_existent, 'foo') is None


@pytest.fixture
def abc_topdir(tmp_path):
    a = tmp_path / 'a'
    b = tmp_path / 'b'
    c = tmp_path / 'c'
    for dir_ in (a, b, c):
        dir_.mkdir()
        for n in range(1, 4):
            f = dir_ / '{}.txt'.format(n)
            f.write_text(str(n))
    return tmp_path


def test_make_directory_structure_nested(abc_topdir):
    paths = [
        str(abc_topdir / 'a' / '1.txt'),
        str(abc_topdir / 'a' / '2.txt'),
        str(abc_topdir / 'a' / '3.txt'),
        str(abc_topdir / 'c' / '1.txt'),
        str(abc_topdir / 'c' / '2.txt'),
    ]
    link_dir = abc_topdir / 'links'
    make_dir_structure(paths,
                       link_dir=link_dir,
                       flatten=False,
                       ref_src_dir=str(abc_topdir))

    for p in paths:
        relpath = os.path.relpath(p, start=abc_topdir)
        f = link_dir / relpath
        assert f.is_symlink()
        assert str(f.resolve().absolute()) == p
    assert not (link_dir / 'c' / '3.txt').exists()
    assert not (link_dir / 'b' / '1.txt').exists()


def test_make_directory_structure_flat(abc_topdir):
    paths = [
        str(abc_topdir / 'a' / '1.txt'),
        str(abc_topdir / 'b' / '2.txt'),
        str(abc_topdir / 'c' / '3.txt'),
    ]
    copy_dir = abc_topdir / 'copies'
    make_dir_structure(paths,
                       copy_dir=copy_dir,
                       flatten=True)

    for n in range(1, 4):
        f = copy_dir / '{}.txt'.format(n)
        assert f.read_text() == str(n)
        assert not f.is_symlink()
    assert not (copy_dir / 'a' / '1.txt').exists()
    assert not (copy_dir / 'b' / '2.txt').exists()
    assert not (copy_dir / 'c' / '3.txt').exists()


def test_make_directory_structure_no_dst():
    with pytest.raises(ValueError):
        make_dir_structure('paths irrelevant')


def test_make_directory_structure_no_ref_src_dir():
    with pytest.raises(ValueError):
        make_dir_structure('paths irrelevant', link_dir='irrelevant',
                           flatten=False)


def test_parse_list():
    assert parse_list('foo,bar,baz') == ['foo', 'bar', 'baz']
    assert parse_list(' foo ,bar ,baz') == ['foo', 'bar', 'baz']
    assert parse_list(' foo SEPbar SEPbaz', sep='SEP') == ['foo', 'bar', 'baz']
    assert parse_list('foo,,,bar,baz') == ['foo', '', '', 'bar', 'baz']
    assert parse_list(',') == ['', '']

