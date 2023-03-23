# -*- coding: UTF-8 -*-
import subprocess
import pathlib
import click
import multiprocessing
import argparse
import os


def transfer_tiatoolbox(source_input, target_input, output_dir, file_types):
    sub_p = subprocess.Popen(
        'tiatoolbox stainnorm --source_input {} --target_input {} '
        ' --method "macenko" --output_dir {} --file_types {}'.format(
            source_input, target_input, output_dir, file_types),
        shell=True,
        stderr=subprocess.PIPE
    )
    sub_p.wait()

    return True


def batch_cn(source_input, target_input, output_dir, file_types='*orig.png'):
    """External package tiatoolbox,
       multi-threaded thread pool automatically completes stainnorm.
    """
    if pathlib.Path(source_input).is_dir():
        paths = pathlib.Path(source_input).glob('*')
        print(multiprocessing.cpu_count())
        pool_multi = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        for path in paths:
            # name = '-'.join(path.stem.split('-')[:3])
            name = path.stem
            pathlib.Path(output_dir).joinpath(name).mkdir(parents=True, exist_ok=True)
            if len(os.listdir(os.path.join(output_dir, name))) > 0:
                continue
            pool_multi.apply(transfer_tiatoolbox,
                             (path,
                              target_input,
                              pathlib.Path(output_dir).joinpath(name),
                              file_types))

        pool_multi.close()
        pool_multi.join()
    elif pathlib.Path(source_input).is_file():
        name = '-'.join(pathlib.Path(source_input).parent.stem.split('-')[:3])
        pathlib.Path(output_dir).joinpath(name).mkdir(parents=True, exist_ok=True)
        transfer_tiatoolbox(pathlib.Path(source_input), target_input,
                            pathlib.Path(output_dir).joinpath(name), file_types)
    else:
        raise FileNotFoundError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tils to colorNormalization')
    parser.add_argument('--source_input', type=str, default="/tmp/data/lung/images")
    parser.add_argument('--target_input', type=str, default="./asset/floderview.png")
    parser.add_argument('--output_dir', type=str, default='/tmp/data/tiles_color_normalized')
    parser.add_argument('--file_types', type=str, default="*orig.png")
    args = parser.parse_args()
    # open('log.txt','a').write(args.output_dir)
    batch_cn(args.source_input, args.target_input, args.output_dir, args.file_types)
