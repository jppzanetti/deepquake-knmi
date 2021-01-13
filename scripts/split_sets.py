import os
import glob
import random
import argparse

def move(src_path, dest_path):
    try:
        os.rename(src_path, dest_path)
    except FileNotFoundError:
        pass


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True,
                    help='Path to the input directory')
parser.add_argument('-o', '--output', type=str, required=True,
                    help='Path to the output directory')
parser.add_argument('--extra_samples', type=int, default=0,
                    help='Number of extra samples generated for each event')
args = parser.parse_args()

# Directory names
SOURCE_DIR = args.input
TRAIN_DIR = os.path.join(args.output, 'train/')
EVAL_DIR = os.path.join(args.output, 'eval/')
TEST_DIR = os.path.join(args.output, 'test/')

NUM_AUGM = args.extra_samples

# Create output directories if needed
for dirname in [TRAIN_DIR, EVAL_DIR, TEST_DIR]:
    if not os.path.exists(dirname):
        os.makedirs(dirname)

basenames = [filepath.split('/')[-1].split('.')[0]
             for filepath
             in glob.glob(os.path.join(SOURCE_DIR, '*[0-9][0-9].tfrecords'))]
random.shuffle(basenames)

n = len(basenames)
i = 0

while i < 0.7 * n:
    basename = basenames[i]
    src_path = os.path.join(SOURCE_DIR, basename + '.tfrecords')
    dest_path = os.path.join(TRAIN_DIR, basename + '.tfrecords')
    move(src_path, dest_path)
    for j in range(NUM_AUGM):
        src_path = os.path.join(SOURCE_DIR, basename + '_' + str(j) + '.tfrecords')
        dest_path = os.path.join(TRAIN_DIR, basename + '_' + str(j) + '.tfrecords')
        move(src_path, dest_path)
    i += 1

while i < 0.85 * n:
    basename = basenames[i]
    src_path = os.path.join(SOURCE_DIR, basename + '.tfrecords')
    dest_path = os.path.join(EVAL_DIR, basename + '.tfrecords')
    move(src_path, dest_path)
    for j in range(NUM_AUGM):
        src_path = os.path.join(SOURCE_DIR, basename + '_' + str(j) + '.tfrecords')
        dest_path = os.path.join(EVAL_DIR, basename + '_' + str(j) + '.tfrecords')
        move(src_path, dest_path)
    i += 1

while i < n:
    basename = basenames[i]
    src_path = os.path.join(SOURCE_DIR, basename + '.tfrecords')
    dest_path = os.path.join(TEST_DIR, basename + '.tfrecords')
    move(src_path, dest_path)
    for j in range(NUM_AUGM):
        src_path = os.path.join(SOURCE_DIR, basename + '_' + str(j) + '.tfrecords')
        dest_path = os.path.join(TEST_DIR, basename + '_' + str(j) + '.tfrecords')
        move(src_path, dest_path)
    i += 1
