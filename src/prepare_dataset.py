import argparse
import ffmpy


def parse_args():
    parser = argparse.ArgumentParser(description='Prepares a dataset for training.')
    parser.add_argument('source', help='source video')
    parser.add_argument('destination', help='destination directory')
    return parser.parse_args()


def main(args):
    ffmpy.FFmpeg(
        inputs={args.source: None},
        outputs={'{}/frame_%03d.png'.format(args.destination): None},
    ).run()


if __name__ == '__main__':
    main(parse_args())
