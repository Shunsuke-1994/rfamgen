import h5py 
import os 
from sklearn.model_selection import train_test_split

def split_onehot_cm(path_to_cmonehot, train_ratio = 0.7, suffix = "", random_state = 42):
    """
    train/valid/test splitter for datasets of CM-VAE.
    """

    h5 = h5py.File(path_to_cmonehot, "r")
    ids = h5["id"][:]
    tr = h5["tr"][:]
    s = h5["s"][:]
    p = h5["p"][:]
    id_train, id_vt,   tr_train,  tr_vt,   s_train,  s_vt,   p_train,  p_vt   = train_test_split(ids, tr, s,p, test_size=1-train_ratio, random_state = random_state)
    id_valid, id_test, tr_valid,  tr_test, s_valid,  s_test, p_valid,  p_test = train_test_split(id_vt, tr_vt, s_vt, p_vt, test_size=0.5, random_state = random_state)

    basename, _ = os.path.splitext(path_to_cmonehot)
    with h5py.File(basename + f"_train{suffix}.h5", "w") as h5:
        h5.create_dataset("id", data = id_train)
        h5.create_dataset("tr", data = tr_train)
        h5.create_dataset("s",  data = s_train)
        h5.create_dataset("p",  data = p_train)

    with h5py.File(basename + f"_valid{suffix}.h5", "w") as h5:
        h5.create_dataset("id", data = id_valid)
        h5.create_dataset("tr", data = tr_valid)
        h5.create_dataset("s",  data = s_valid)
        h5.create_dataset("p",  data = p_valid)

    with h5py.File(basename + f"_test{suffix}.h5", "w") as h5:
        h5.create_dataset("id", data = id_test)
        h5.create_dataset("tr", data = tr_test)
        h5.create_dataset("s",  data = s_test)
        h5.create_dataset("p",  data = p_test)

    return basename + "_(train|valid|test).h5"

def split_onehot_cg(path_to_onehot, train_ratio = 0.7, suffix = "", random_state = 42):
    """
    train/valid/test splitter for datasets of Character/Garmmar/GapChar-VAE.
    """
    h5 = h5py.File(path_to_onehot, "r")
    ids = h5["id"][:]
    onehot = h5["data"][:]
    id_train, id_vt, onehot_train, onehot_tv = train_test_split(ids, onehot, test_size=1-train_ratio, random_state=random_state)
    id_valid, id_test, onehot_valid, onehot_test = train_test_split(id_vt, onehot_tv, test_size=0.5, random_state=random_state)
    
    basename, _ = os.path.splitext(path_to_onehot)
    with h5py.File(basename + f"_train{suffix}.h5", "w") as h5:
        h5.create_dataset("id", data = id_train)
        h5.create_dataset("data", data = onehot_train)

    with h5py.File(basename + f"_valid{suffix}.h5", "w") as h5:
        h5.create_dataset("id", data = id_valid)
        h5.create_dataset("data", data = onehot_valid)

    with h5py.File(basename + f"_test{suffix}.h5", "w") as h5:
        h5.create_dataset("id", data = id_test)
        h5.create_dataset("data", data = onehot_test)

    return basename + "_(train|valid|test).h5"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default = "./datasets/ForDMSdata/RF00028/RF00028_unique_seed_removed_244ntTO304nt_refoldSScons_g3parsable_seqonly_notrunc_traceback_onehot_cm.h5", type = str)
    parser.add_argument('--suffix', default = "")
    parser.add_argument("--train_ratio", default = 0.7, type = float)
    parser.add_argument("--random_state", default = 42, type = int)
    args = parser.parse_args()

    if "onehot_cm" in args.input:
        outfiles = split_onehot_cm(args.input, train_ratio = args.train_ratio, suffix = args.suffix, random_state = args.random_state)
        print("Splited into", outfiles)

    elif ("onehot_g" in args.input) or ("onehot_nuc" in args.input):
        outfiles = split_onehot_cg(args.input, train_ratio = args.train_ratio, suffix = args.suffix, random_state = args.random_state)
        print("Splited into", outfiles)
    else:
        print("Input file has to contain 'onehot_(nuc|g|cm).'")
    
