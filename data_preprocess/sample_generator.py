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

args = parser.parse_args()

if __name__ == '__main__':
    src_sentences = []
    tgt_sentences = []
    src_writer = open(os.path.join(args.data_dir, "sentence-pair-src-{0}".format(args.suffix)), 'w')
    tgt_writer = open(os.path.join(args.data_dir, "sentence-pair-tgt-{0}".format(args.suffix)), 'w')
    label_writer = open(os.path.join(args.data_dir, "sentence-pair-label-{0}".format(args.suffix)), 'w')
    with open(os.path.join(args.data_dir, args.src), 'r') as src_reader,\
        open(os.path.join(args.data_dir, args.tgt), 'r') as tgt_reader:
        for src_line, tgt_line in zip(src_reader.readlines(), tgt_reader.readlines()):
            if args.ratio < np.random.uniform():
                src_writer.write(src_line)
                tgt_writer.write(tgt_line)
                label_writer.write("1")
                label_writer.write("\n")
            else:
                src_sentences.append(src_line)
                tgt_sentences.append(tgt_line)

    assert len(src_sentences) == len(tgt_sentences)

    total_negative_samples = len(src_sentences)

    for i, each in enumerate(src_sentences):
        index = np.random.randint(0, total_negative_samples)
        while index == i:
            index = np.random.randint(0, total_negative_samples)
        src_writer.write(each)
        tgt_writer.write(tgt_sentences[index])
        label_writer.write("0")
        label_writer.write("\n")

    src_writer.close()
    tgt_writer.close()
    label_writer.close()
