import tensorflow as tf
import os
from tensor2tensor.data_generators import generator_utils
from SpmTextEncoder import SpmTextEncoder
flags = tf.flags

FLAGS = flags.FLAGS


flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string("vocab", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "tmp_dir", None,
    "The output directory where the model checkpoints will be written.")


flags.DEFINE_string(
    "src", None,
    "src suffix for vocab.")

flags.DEFINE_string(
    "tgt", None,
    "tgt suffix for vocab.")


flags.DEFINE_integer(
    "max_seq_length", 224,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")


def example_generator(dir_name, tag="train"):
    src_encoder = SpmTextEncoder(os.path.join(FLAGS.data_dir, FLAGS.vocab + "." + FLAGS.src + ".model"))
    tgt_encoder = SpmTextEncoder(os.path.join(FLAGS.data_dir, FLAGS.vocab + "." + FLAGS.tgt + ".model"))
    with open(os.path.join(dir_name, "sentence-pair-src-{0}".format(tag)), "r") as src, \
        open(os.path.join(dir_name, "sentence-pair-tgt-{0}".format(tag)), "r") as tgt, \
        open(os.path.join(dir_name, "sentence-pair-label-{0}".format(tag)), "r") as labels:
        for src_line, tgt_line, label in zip(src.readlines(), tgt.readlines(), labels.readlines()):
            feature = dict()
            feature["src_ids"] = src_encoder.encode(src_line.strip())[0: FLAGS.max_seq_length]
            feature["tgt_ids"] = tgt_encoder.encode(tgt_line.strip())[0: FLAGS.max_seq_length]
            feature["label"] = [int(label.strip())]
            yield feature


def main(_):
    train_shards = 100
    dev_shards = 1
    train_file_names = [os.path.join(FLAGS.data_dir, "{0}-{1}-train-000{2}-of-00{3}"
                                     .format(FLAGS.src, FLAGS.tgt, i, train_shards))
                        for i in range(train_shards)]

    dev_file_names = [os.path.join(FLAGS.data_dir, "{0}-{1}-dev-000{2}-of-00{3}"
                                   .format(FLAGS.src, FLAGS.tgt, i, dev_shards))
                      for i in range(dev_shards)]

    train_generator = example_generator(FLAGS.tmp_dir, "train")

    eval_generator = example_generator(FLAGS.tmp_dir, "dev")

    generator_utils.generate_files(train_generator, train_file_names, cycle_every_n=10)

    generator_utils.generate_files(eval_generator, dev_file_names, cycle_every_n=10)

    generator_utils.shuffle_dataset(train_file_names)

    generator_utils.shuffle_dataset(dev_file_names)


if __name__ == '__main__':
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("src")
    flags.mark_flag_as_required("tgt")
    flags.mark_flag_as_required("vocab")
    flags.mark_flag_as_required("tmp_dir")
    tf.app.run()
