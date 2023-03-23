import argparse
import openslide
import os
import sys
from glob import glob
from os.path import join
from preprocessing.load_xml import load_xml
from preprocessing.get_ret_ifo import get_ret_ifo
from utils.common import logger
import multiprocessing


path_wd = os.path.dirname(sys.argv[0])
sys.path.append(path_wd)
if not path_wd == '':
    os.chdir(path_wd)
need_save = False


def multiprocessing_segmentation(xml, index, images_dir_split, size_square, prepare_type):
    xy_list = load_xml(xml[index])
    if os.path.exists(xml[index].split("xml")[0]+prepare_type):
        image_address = xml[index].split("xml")[0] + prepare_type
        slide = openslide.open_slide(image_address)
        # image_large = \
        get_ret_ifo(xy_list, slide, image_address, images_dir_split,
                    size_square, size_square, 3, 0.3)


def prepare_data(images_dir_root, images_dir_split, size_square, prepare_type):
    num_name = 0

    image_dir_list = glob(join(images_dir_root, r'*/'))
    segmentation_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for image_dir in image_dir_list:
        xml_files = glob(join(image_dir, '*.xml'))
        if len(xml_files) == 0:
            # raise FileNotFoundError
            continue
        for index_xml in range(len(xml_files)):
            num_name += 1
            logger.info("xml_files: {}".format(xml_files[index_xml]))

            segmentation_pool.apply(multiprocessing_segmentation,
                                    (xml_files, index_xml, images_dir_split, size_square, prepare_type))

    logger.info('tiles are done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='svs to tiles')
    parser.add_argument('--slide_image_root', type=str, default="/tmp/data/lung/images")
    parser.add_argument('--tiles_image_root', type=str, default="/tmp/data/tiles")
    parser.add_argument('--size_square', type=int, default=512)
    parser.add_argument('--prepare_types', type=str, default="svs")
    args = parser.parse_args()

    logger.info('Processing svs images to tiles')
    available_policies = ["svs", "ndpi"]
    assert args.prepare_types in available_policies, "svs or ndpi slide support only"
    prepare_data(args.slide_image_root, args.tiles_image_root, args.size_square, args.prepare_types)
