import os
import re
import configparser


def _label_finder(str):
    Om_label = re.search(r"(?<=Om=).+(?=_eta)", str).group(0)
    s8_label = re.search(r"(?<=s8=).+(?=_tomo)", str).group(0)
    return (float(Om_label), float(s8_label))


if __name__ == "__main__":
    path = "/cluster/work/refregier/besuter/data/LIGHTCONES"
    config = configparser.ConfigParser()
    config.add_section("z=0.0_0 Om")
    config.add_section("z=0.0_0 s8")
    lightcone_files = [
        file for file in os.listdir(path)
        if file.endswith("tomo=1_z=0.0_0.fits") and file.startswith("LIGHTCONE")
    ]
    for idx, file in enumerate(lightcone_files):
        labels = _label_finder(file)
        config.set("z=0.0_0 Om", str(idx+1), str(labels[0]))
        config.set("z=0.0_0 s8", str(idx + 1), str(labels[1]))

    with open("LIGHTCONE.ini", "w") as configfile:
        config.write(configfile)