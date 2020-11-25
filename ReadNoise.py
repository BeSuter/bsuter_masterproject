import os
import numpy as np
import configparser

if __name__ == "__main__":
    path = "/cluster/work/refregier/besuter/master_branch/data/STATS/Noise"
    config = configparser.ConfigParser()
    for tomo in range(4):
        config.add_section(str(tomo+1))
        noise_files = [
            os.path.join(path, file) for file in os.listdir(path)
            if file.endswith(f"tomo={str(tomo+1)}x{str(tomo+1)}.npy")
        ]
        std = 0.0
        mean = 0.0
        for file in noise_files:
            std_all_cuts = np.sum(np.sqrt(np.load(file)[:, 1][11::12].astype(float)))
            mean_all_cuts = np.sum(np.load(file)[:, 2][11::12].astype(float))
            std += std_all_cuts
            mean += mean_all_cuts
        std /= 8.0 * len(noise_files)
        mean /= 8. * len(noise_files)
        config.set(str(tomo+1), "std", str(std))
        config.set(str(tomo+1), "mean", str(mean))

    with open("noise.ini", "w") as configfile:
        config.write(configfile)