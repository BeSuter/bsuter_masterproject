import os
import configparser

if __name__ == "__main__":
    path = "/cluster/work/refregier/besuter/data/LIGHTCONES"
    config = configparser.ConfigParser()
    config.add_section("z=0.0_0")
    lightcone_files = [
        file for file in os.listdir(path)
        if file.endswith("z=0.0_0.fits.npy") and file.startswith("LIGHTCONE")
    ]
    for idx, file in enumerate(lightcone_files):
        config.set("z=0.0_0", str(idx), file)

    with open("noise.ini", "w") as configfile:
        config.write(configfile)