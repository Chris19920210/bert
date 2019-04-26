import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(description='sample generate')

parser.add_argument('--data-dir', type=str, default=None,
                    help='data dir')
parser.add_argument('--src', type=str, default=None,
                    help='src language')
parser.add_argument('--tgt', type=str, default=None,
                    help='tgt language')
parser.add_argument('--ratio', type=float, default=0.5,
                    help='ratio for positive samples')
parser.add_argument('--suffix', type=str, default=None,
                    help='suffix for generated data')
parser.add_argument('--window', type=int, default=100,
                    help='window for generate negative samples')
parser.add_argument('--hard-ratio', type=float, default=0.5,
                    help='hard negative samples ratio')
parser.add_argument('--output-dir', type=str, default=None,
                    help='the output dir for data')

args = parser.parse_args()

if __name__ == '__main__':
    src_sentences = []
    tgt_sentences = []
    if args.output_dir is None:
        output_dir = args.data_dir
    else:
        output_dir = args.output_dir

    src_writer = open(os.path.join(output_dir, "sentence-pair-src-{0}".format(args.suffix)), 'w')
    tgt_writer = open(os.path.join(output_dir, "sentence-pair-tgt-{0}".format(args.suffix)), 'w')
    label_writer = open(os.path.join(output_dir, "sentence-pair-label-{0}".format(args.suffix)), 'w')
    with open(os.path.join(args.data_dir, args.src), 'r') as src_reader,\
        open(os.path.join(args.data_dir, args.tgt), 'r') as tgt_reader:
        for src_line, tgt_line in zip(src_reader.readlines(), tgt_reader.readlines()):
            src_sentences.append(src_line)
            tgt_sentences.append(tgt_line)

    width = args.window // 2

    for i, each in enumerate(src_sentences):
        src_writer.write(each)
        if args.ratio < np.random.uniform():
            index = i
            label = "1"
        else:
            if args.hard_ratio > np.random.uniform():
                left, right = 0, len(src_sentences)
            else:
                left, right = max(0, i - width), min(i + width, len(src_sentences))
            index = np.random.randint(left, right)
            while index == i:
                index = np.random.randint(left, right)
            label = "0"
        tgt_writer.write(tgt_sentences[index])
        label_writer.write(label)
        label_writer.write('\n')

    src_writer.close()
    tgt_writer.close()
    label_writer.close()
