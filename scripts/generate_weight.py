import h5py
import numpy as np
import time 
import random
random.seed(42)
from multiprocessing import Pool

# compute abs distance among dataset
def load_data_cm(path):
    """
    (N_SAMPLES, N_RULES, LENGTH)
    """
    data = h5py.File(path, "r")
    tr = np.nan_to_num(data["tr"][:], copy = False).transpose(0, -1, -2)
    s = data["s"][:].transpose(0, -1, -2)
    p = data["p"][:].transpose(0, -1, -2)
    return tr, s, p

# compute abs distance among dataset
def load_data_cg(path):
    """
    (N_SAMPLES, N_RULES, LENGTH)
    """
    data = h5py.File(path, "r")
    onehot = data["data"][:].transpose(0, -1, -2)
    return onehot

def calc_distance_cm(trsp_i, trsp_j):
    (tr_i, s_i, p_i), (tr_j, s_j, p_j) = trsp_i, trsp_j
    distance = np.abs(tr_i - tr_j).sum() + np.abs(s_i - s_j).sum() + np.abs(p_i - p_j).sum()
    return distance

def calc_distance_cg(xi, xj):
    distance = np.abs(xi - xj).sum()
    return distance

def compute_and_write_weight(X_fname, threshold, outfile, mode = "cm", sampling_threshold = 10000, sample_ratio_over_threshold = 0.05, cpu = 4, print_every = 500):
    assert args.mode in {"cm", "c", "g"}, "Select mode from c/g/cm"
    
    if mode == "cm":
        tr_train, s_train, p_train = load_data_cm(X_fname)
        N_COLLUMNS = tr_train.shape[-1] + s_train.shape[-1] + p_train.shape[-1]
        Ntotal      = tr_train.shape[0]
        if Ntotal < sampling_threshold: 
            n_samples = Ntotal
        else:
            n_samples = int(Ntotal*sample_ratio_over_threshold)
            
        print("CM mode.")
        print(f"Sampled size: {str(n_samples)}")
        print(f"N_COLUMNS: {str(N_COLLUMNS)}")
        print(f"*"*50)
        Neff = 0
        with h5py.File(outfile, "w") as f:
            f.create_dataset('weight', (Ntotal, ))
            start = time.time()
            for i in range(Ntotal):
                tmp_index = set(range(Ntotal))
                tmp_index.remove(i)
                sampled_index = random.sample(tmp_index, k = int(n_samples -1))
                tr_i, s_i, p_i = tr_train[i], s_train[i], p_train[i]
                onehot_pairs = [((tr_i, s_i, p_i), (tr_train[j], s_train[j], p_train[j])) for j in sampled_index]

                with Pool(cpu) as p:  
                    distances = p.starmap(calc_distance_cm, onehot_pairs)

                n_neighbor = 1 + sum([1 if d < N_COLLUMNS*threshold else 0 for d in distances])
                weight = 1/n_neighbor
                f["weight"][i] = weight
                Neff += weight
                
                if i%print_every == 0:
                    finish = time.time()
                    print(f"{i}/{str(Ntotal)}\t, sampling {str(n_samples)}, estimated remaining time:", (Ntotal - i)*(finish - start)/print_every, "sec.")
                    start = time.time()

    elif mode in "cg":
        onehot = load_data_cg(X_fname)
        N_COLLUMNS  = onehot.shape[-1]
        Ntotal      = onehot.shape[0]
        if Ntotal < sampling_threshold: 
            n_samples = Ntotal
        else:
            n_samples = int(Ntotal*sample_ratio_over_threshold)
        print(f"Char/Gram mode.")
        print(f"Sampled size: {str(n_samples)}")
        print(f"N_COLUMNS: {str(N_COLLUMNS)}")
        print(f"*"*50)
        Neff = 0
        with h5py.File(outfile, "w") as f:
            f.create_dataset('weight', (Ntotal, ))
            start = time.time()
            for i in range(Ntotal):
                tmp_index = set(range(Ntotal))
                tmp_index.remove(i)
                sampled_index = random.sample(tmp_index, k = int(n_samples -1))
                x_i = onehot[i]
                onehot_pairs = [(x_i, onehot[j]) for j in sampled_index]

                with Pool(cpu) as p:
                    distances = p.starmap(calc_distance_cg, onehot_pairs)

                n_neighbor = 1 + sum([1 if d < N_COLLUMNS*threshold else 0 for d in distances])
                weight = 1/n_neighbor
                f["weight"][i] = weight
                Neff += weight
                if i%print_every == 0:
                    finish = time.time()
                    print(f"{i}/{str(Ntotal)}\t, sampling {str(n_samples)}, estimated remaining time:", (Ntotal - i)*(finish - start)/print_every, "sec.")
                    start = time.time()

    return Ntotal, Neff


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default = "cm", type = str)
    parser.add_argument('-i', '--input', default = "./datasets/ForDMSdata/RF00028/RF00028_unique_seed_removed_244ntTO304nt_refoldSScons_g3parsable_seqonly_notrunc_traceback_onehot_cm.h5", type = str)
    parser.add_argument('-o', '--output', default = "")
    parser.add_argument("--threshold", default = 0.2, type = float)
    parser.add_argument("--n_samples", default = float('inf'), type = float)
    parser.add_argument("--cpu", default = 8, type = int)
    parser.add_argument("--print_every", default = 500, type = int)
    args = parser.parse_args()


    if args.output == "":
        tmp = str(args.threshold).replace(".", "p")
        outfile = os.path.splitext(args.input)[0] + f"_weight_threshold{tmp}.h5"
    else:
        outfile = args.output

    print(f"Infile\t\t: {args.input}")
    print(f"mode\t\t: {args.mode}")
    print(f"n_samples\t: {args.n_samples}")
    print(f"print_every\t: {args.print_every}")
    print(f"Outfile\t\t: {outfile}")
    print(f"Threshold\t: {args.threshold}")
    print(f"N_CORE\t: {args.cpu}")
    print(f"*"*50)

    Ntotal, Neff = compute_and_write_weight(
        X_fname = args.input, 
        threshold = args.threshold,
        outfile = outfile,
        mode = args.mode,
        sampling_threshold = args.n_samples,
        cpu = args.cpu, 
        print_every = args.print_every
        )
    
    print(f"Total data size\t\t: {str(Ntotal)} seq")
    print(f"Effective data size\t: {str(Neff)} seq")
    print(f"Saved {args.output}")
