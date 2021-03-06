import os
import re

import healpy as hp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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



def _shape_finder(str):
    shapes = re.search(r"(?<=shapes=)\d+,\d+&\d+(?=_)", str).group(0)
    all_shapes = []
    for item in shapes.split("&"):
        try:
            all_shapes.append((int(item.split(",")[0]), int(item.split(",")[1])))
        except IndexError:
            all_shapes.append((int(item)))
    return all_shapes


def get_dataset(path=[]):
    if not isinstance(path, list):
        path = [path]
    all_files = []
    for pp in path:
        f_names = [
            os.path.join(pp, file) for file in os.listdir(pp)
            if file.endswith(".tfrecord")
        ]
        all_files.extend(f_names)
    shapes = _shape_finder(all_files[0])
    dset = tf.data.TFRecordDataset(all_files)
    decoded_dset = decode_labeled_dset(dset, shapes)
    return decoded_dset

if __name__ == "__main__":
    fid_dir = "/scratch/snx3000/bsuter/TFRecordFiducial"
    noise_dir = "/scratch/snx3000/bsuter/TFRecordNoise"

    map_save_dir = "/users/bsuter/maps_for_janis"
    os.makedirs(map_save_dir, exist_ok=True)
    
    count = 0
    count_count = 0
    final_res = {"smoothed": {}, "double_smoothed": {}, "gauss_CL": {}}
    store_indices = {"smoothed": {}, "double_smoothed": {}}
    check_indices = {"smoothed": [], "double_smoothed": []}
    for fid_maps, noise_maps in zip(get_dataset(fid_dir), get_dataset(noise_dir)):
        count += 1
        count_count += 1


        fiducial_map_1 = fid_maps[0][0]
        noise_map_1 = noise_maps[0][0]

        print("Tomo 1")
        print("Map")
        print(np.arange(len(fiducial_map_1))[fiducial_map_1 > hp.UNSEEN])
        print("Noise")
        print(np.arange(len(noise_map_1))[noise_map_1 > hp.UNSEEN])
        print("Combined")
    
        """fpp1 = hp.anafast(fiducial_map_1.numpy())
        npp1 = hp.anafast(noise_map_1.numpy())
        plt.figure()
        plt.loglog(fpp1, label="Only double smoothed fiducial")
        plt.loglog(npp1, label="Only double smoothed noise")
        plt.legend()
        plt.savefig("/users/bsuter/Compare_PP/Single_Double_Smoothed.png")"""
        
        """hp.mollview(fiducial_map_1.numpy(), nest=True, title="Double Smoothed Fiducial Map")
        plt.savefig("/users/bsuter/Compare_PP/fiducial_map_1.png")
    
        hp.mollview(noise_map_1.numpy(), nest=True, title="Double Smoothed Noise Map")
        plt.savefig("/users/bsuter/Compare_PP/noise_map_1.png")"""
    
        mask = fiducial_map_1.numpy() < hp.UNSEEN
        full_double_smoothed_1 = fiducial_map_1.numpy() + noise_map_1.numpy()
        full_double_smoothed_1[mask] = hp.UNSEEN
        only_network_input = full_double_smoothed_1[full_double_smoothed_1 > hp.UNSEEN]
        np.save(os.path.join(map_save_dir, f"TFRecord_Full_Map_{count}_tomo={1}"), only_network_input)
        indices_tfr = np.arange(len(full_double_smoothed_1))[full_double_smoothed_1 > hp.UNSEEN]
        print(indices_tfr)
        print("\n")
        # np.save(os.path.join(map_save_dir, f"TFRecord_Indices"), indices_tfr)
        """hp.mollview(full_double_smoothed_1, nest=True, title="Full Double Smoothed map")
        plt.savefig("/users/bsuter/Compare_PP/full_double_smoothed_1.png")"""
        try:
            check_indices["double_smoothed"].append(store_indices["double_smoothed"][1] == indices_tfr)
        except KeyError:
            store_indices["double_smoothed"][1] = indices_tfr
            check_indices["double_smoothed"].append(store_indices["double_smoothed"][1] == indices_tfr)

        full_double_smoothed_1 = hp.reorder(full_double_smoothed_1, n2r=True)
        try:
            final_res["double_smoothed"][1] += hp.anafast(full_double_smoothed_1)
        except KeyError:
            final_res["double_smoothed"][1] = hp.anafast(full_double_smoothed_1)
    
        """plt.figure()
        plt.loglog(pp_double_smoothed_1, label="Full Double Smoothed 1")
        plt.legend()
        plt.savefig("/users/bsuter/Compare_PP/only_DoubleSmoothedPP.png")"""
    
        fiducial_map_2 = fid_maps[0][1]
        noise_map_2 = noise_maps[0][1]

        print("Tomo 2")
        print("Map")
        print(np.arange(len(fiducial_map_2))[fiducial_map_2 > hp.UNSEEN])
        print("Noise")
        print(np.arange(len(noise_map_2))[noise_map_2 > hp.UNSEEN])
        print("Combined")
    
        mask = fiducial_map_2.numpy() < hp.UNSEEN
        full_double_smoothed_2 = fiducial_map_2.numpy() + noise_map_2.numpy()
        full_double_smoothed_2[mask] = hp.UNSEEN
        full_double_smoothed_2 = hp.reorder(full_double_smoothed_2, n2r=True)
        only_network_input = full_double_smoothed_2[full_double_smoothed_2 > hp.UNSEEN]
        # np.save(os.path.join(map_save_dir, f"TFRecord_Full_Map_{count}_tomo={2}"), only_network_input)
        indices_tfr = np.arange(len(full_double_smoothed_2))[full_double_smoothed_2 > hp.UNSEEN]
        print(indices_tfr)
        print("\n")
        check_indices["double_smoothed"].append(store_indices["double_smoothed"][1] == indices_tfr)

        try:
            final_res["double_smoothed"][2] += hp.anafast(full_double_smoothed_2)
        except KeyError:
            final_res["double_smoothed"][2] = hp.anafast(full_double_smoothed_2)
    
        fiducial_map_3 = fid_maps[0][2]
        noise_map_3 = noise_maps[0][2]

        print("Tomo 3")
        print("Map")
        print(np.arange(len(fiducial_map_3))[fiducial_map_3 > hp.UNSEEN])
        print("Noise")
        print(np.arange(len(noise_map_3))[noise_map_3 > hp.UNSEEN])
        print("Combined")
    
        mask = fiducial_map_3.numpy() < hp.UNSEEN
        full_double_smoothed_3 = fiducial_map_3.numpy() + noise_map_3.numpy()
        full_double_smoothed_3[mask] = hp.UNSEEN
        full_double_smoothed_3 = hp.reorder(full_double_smoothed_3, n2r=True)
        only_network_input = full_double_smoothed_3[full_double_smoothed_3 > hp.UNSEEN]
        # np.save(os.path.join(map_save_dir, f"TFRecord_Full_Map_{count}_tomo={3}"), only_network_input)
        indices_tfr = np.arange(len(full_double_smoothed_3))[full_double_smoothed_3 > hp.UNSEEN]
        print(indices_tfr)
        print("\n")
        check_indices["double_smoothed"].append(store_indices["double_smoothed"][1] == indices_tfr)

        try:
            final_res["double_smoothed"][3] += hp.anafast(full_double_smoothed_3)
        except KeyError:
            final_res["double_smoothed"][3] = hp.anafast(full_double_smoothed_3)
    
        fiducial_map_4 = fid_maps[0][3]
        noise_map_4 = noise_maps[0][3]

        print("Tomo 4")
        print("Map")
        print(np.arange(len(fiducial_map_3))[fiducial_map_3 > hp.UNSEEN])
        print("Noise")
        print(np.arange(len(noise_map_3))[noise_map_3 > hp.UNSEEN])
        print("Combined")
    
        mask = fiducial_map_4.numpy() < hp.UNSEEN
        full_double_smoothed_4 = fiducial_map_4.numpy() + noise_map_4.numpy()
        full_double_smoothed_4[mask] = hp.UNSEEN
        full_double_smoothed_4 = hp.reorder(full_double_smoothed_4, n2r=True)
        only_network_input = full_double_smoothed_4[full_double_smoothed_4 > hp.UNSEEN]
        # np.save(os.path.join(map_save_dir, f"TFRecord_Full_Map_{count}_tomo={4}"), only_network_input)
        indices_tfr = np.arange(len(full_double_smoothed_4))[full_double_smoothed_4 > hp.UNSEEN]
        print(indices_tfr)
        print("\n")
        check_indices["double_smoothed"].append(store_indices["double_smoothed"][1] == indices_tfr)

        try:
            final_res["double_smoothed"][4] += hp.anafast(full_double_smoothed_4)
        except KeyError:
            final_res["double_smoothed"][4] = hp.anafast(full_double_smoothed_4)

        dir = "/scratch/snx3000/bsuter/Maps"
        all_ids = np.load(os.path.join(dir, "Map_ids.npy"))
    

        id = all_ids[int(count * -1)]
        for tomo in range(1, 5):
            try:
                map = np.load(os.path.join(dir, "FullMaps", f"Map_Om=0.26_s8=0.84_tomo={tomo}_id={id}.npy"))
                map = hp.reorder(map, n2r=True)
                only_network_input = map[map > hp.UNSEEN]
                # np.save(os.path.join(map_save_dir, f"Pipeline_Map_{count_count}_tomo={tomo}"), only_network_input)
                indices_pipeline = np.arange(len(map))[map > hp.UNSEEN]
                try:
                    check_indices["smoothed"].append(store_indices["smoothed"][1] == indices_pipeline)
                except KeyError:
                    store_indices["smoothed"][1] = indices_pipeline
                    check_indices["smoothed"].append(store_indices["smoothed"][1] == indices_pipeline)
            except FileNotFoundError:
                count_count -= 1
                continue
            ps = hp.anafast(map)

            try:
                final_res["smoothed"][tomo] += ps
            except KeyError:
                final_res["smoothed"][tomo] = ps
        # indices_pipeline = np.arange(len(map))[map > hp.UNSEEN]
        # np.save(os.path.join(map_save_dir, f"Pipeline_Indices"), indices_pipeline)
        print(f"Finished with count={count}")
        if count == 10:
            break
    # print(check_indices)
    np.savez("/users/bsuter/check_indices.npz", **check_indices)

    id = all_ids[-1]
    tomo = 1
    map = np.load(os.path.join(dir, "FullMaps", f"Map_Om=0.26_s8=0.84_tomo={tomo}_id={id}.npy"))
    map = hp.reorder(map, n2r=True)
    mask = map < -1e25
    smoothed_map = hp.sphtfunc.smoothing(map, fwhm=np.radians(float(2.6) / 60.))
    smoothed_map[mask] = hp.UNSEEN
    mCL = hp.anafast(map)
    sCL = hp.anafast(smoothed_map)
    final_res["gauss_CL"] = np.divide(sCL, mCL)

    plt.figure()
    for tomo in range(1, 5):
        final_res["smoothed"][tomo] /= count_count
        final_res["double_smoothed"][tomo] /= count
        plt.plot(np.divide(final_res["double_smoothed"][tomo], final_res["smoothed"][tomo]),
                 label=f"Smoothing CLs for tomo={tomo}")
    plt.plot(final_res["gauss_CL"], label="Gauss CLs")
    plt.title(f"Smoothing CLs averaged over {count} Fiducial Maps")
    plt.legend()
    plt.savefig("/users/bsuter/SmoothingCLs.png")

    plt.figure()
    for tomo in range(1, 5):
        ratio = np.divide(np.divide(final_res["double_smoothed"][tomo], final_res["smoothed"][tomo]),
                          final_res["gauss_CL"])
        plt.plot(ratio, label=f"Ratio for tomo={tomo}")
    plt.title(f"CL ratios")
    plt.legend()
    plt.savefig("/users/bsuter/Ratios.png")

    #np.savez("/users/bsuter/CLs.npz", **final_res)
