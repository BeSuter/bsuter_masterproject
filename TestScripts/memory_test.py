import tensorflow as tf
import psutil


if __name__ == "__main__":
    dataset = tf.data.Dataset.range(int(1e7))
    iterator = dataset.shuffle(int(1e7)).batch(int(1e6))

    print("")
    print("##############################################")
    print("############# STARTING ITERATION #############")

    for _ in iterator:
        used_mem = psutil.virtual_memory().used
        print("used memory: {} Mb".format(used_mem / 1024 / 1024))