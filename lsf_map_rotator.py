import sys
import os
from utils import LSF_map_rotator

if __name__ == "__main__":
    args = sys.argv[1:]
    job_index = str(os.environ[str(args[0])])
    LSF_map_rotator(job_index)