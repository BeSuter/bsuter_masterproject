from utils import tfrecord_writer

if __name__ == "__main__":
    print("Starting TFRecord_writer")
    tfrecord_writer("/scratch/snx3000/bsuter/kappa_maps", downsampling=256)
    print("Done")