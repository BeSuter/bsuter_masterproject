import numpy as np
import healpy as hp
import tensorflow as tf


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.ravel().tolist()))


def extend_indices(indices, nside_in, nside_out, nest=True):
    """
    Minimally extends a set of indices such that it can be reduced to nside_out in a healpy fashion, always four pixels
    reduce naturally to a higher order pixel. Note that this function supports the ring ordering, however, since almost
    no other function does so, nest ordering is strongly recommended.
    :param indices: 1d array of integer pixel ids
    :param nside_in: nside of the input
    :param nside_out: nside of the output
    :param nest: indices are ordered in the "NEST" ordering scheme
    :return: returns a set of indices in the same ordering as the input.
    """
    # figire out the ordering
    if nest:
        ordering = "NEST"
    else:
        ordering = "RING"

    # get the map to reduce
    m_in = np.zeros(hp.nside2npix(nside_in))
    m_in[indices] = 1.0

    # reduce
    m_in = hp.ud_grade(map_in=m_in, nside_out=nside_out, order_in=ordering, order_out=ordering)

    # expand
    m_in = hp.ud_grade(map_in=m_in, nside_out=nside_in, order_in=ordering, order_out=ordering)

    # get the new indices
    return np.arange(hp.nside2npix(nside_in))[m_in > 1e-12]


def serialize_labeled_example(sample, label):
    """
    Create a serialized protobuffer from a given sample and the corresponding label. The inputs will be converted to
    float32.
    :param sample: A numpy array containing the sample
    :param label: A numpy array containing the label
    :return: a serialized protobuffer that can be written to a tfrecord file
    """

    feature = {"sample": _float_feature(sample.ravel().astype(np.float32)),
               "label": _float_feature(label.ravel().astype(np.float32))}

    # create the proto buffer
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def decode_labeled_dset(dset, shapes, auto_tune=True):
    """
    Returns a dataset where the proto bufferes were decoded according to the prescription of serialize_labeled_example
    :param dset: the data set to decode
    :param shapes: a list of shapes [shape_sample, shape_label]
    :param auto_tune: use the experimental auto tune feature for the final mapping (dynamic CPU allocation)
    :return: the decoded dset having two elements, sample and label
    """

    # a function to decode a single proto buffer
    def decoder_func(record_bytes):
        scheme = {"sample": tf.io.FixedLenFeature(shapes[0], dtype=tf.float32),
                  "label": tf.io.FixedLenFeature(shapes[1], dtype=tf.float32)}

        example = tf.io.parse_single_example(
                    # Data
                    record_bytes,
                    # Schema
                    scheme
                  )
        return example["sample"], example["label"]

    # return the new dset
    if auto_tune:
        return dset.map(decoder_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # otherwise serial
    return dset.map(decoder_func)