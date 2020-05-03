from argparse import ArgumentParser
from lib.stitch import stitch_images

def configs():
    parser = ArgumentParser()
    parser.add_argument('ratio', type = int, help = 'Downscaling ratio. ')
    parser.add_argument('f', type = float, help = 'Focal length, in pixels.')
    parser.add_argument('run', type = str, help = 'Testing set to run.')
    parser.add_argument('--use_cache', action = 'store_true', help = 'Use pre-computed features. Advised.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = configs()
    stitch_images(args.run, args.f, args.ratio, args.use_cache)

    