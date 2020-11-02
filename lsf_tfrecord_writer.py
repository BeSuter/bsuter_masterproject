import sys
import os
from utils import LSF_tfrecord_writer

if __name__ == "__main__":
    args = sys.argv[1:]
    job_index = int(os.environ[str(args[0])])
    LSF_tfrecord_writer(job_index)