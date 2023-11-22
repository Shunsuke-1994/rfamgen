# a wrapper script for r2r.
# ref: https://sourceforge.net/projects/weinberg-r2r/

import subprocess
import h5py
from Bio import AlignIO

def append_custom_weight_to_stofile(sto_input, sto_output, weight):
    """
    craete a weighted stockholm.
    sto_input: stockholm
    weight: h5 formart. the order must be same as sto_input.
    see: section 4.3.2 of R2R(1.0.6) manual
    """
    w = h5py.File(weight, "r")["weight"][:]
    ids = list(map(lambda x: x.id, AlignIO.read(sto_input, "stockholm")))
    with open(sto_input, "r") as aln:
        content = aln.readlines()

    USE_THIS_WEIGHT_MAP = []
    for id_, w_ in zip(ids, w):
        USE_THIS_WEIGHT_MAP.append(str(ids))
        USE_THIS_WEIGHT_MAP.append(str(round(w, 6)))
    USE_THIS_WEIGHT_MAP = "#=GF USE_THIS_WEIGHT_MAP " + " ".join(USE_THIS_WEIGHT_MAP) + "\n"
    content[-1] = USE_THIS_WEIGHT_MAP
    content.append("//\n")

    with open(sto_output, "w") as aln_w:
        for line in content:
            aln_w.write(line)

    cmd = f"r2r --GSC-weighted-consensus {sto_input} {sto_output} 3 0.97 0.9 0.75 4 0.97 0.9 0.75 0.5 0.1"
    print(cmd)
    res = subprocess.run(cmd, shell = True, capture_output=True)
    print(res.stdout.decode())
    print(res.stderr.decode())
    return  

def append_GSC_weight_to_stofile(sto_input, sto_output):
    cmd = f"r2r --GSC-weighted-consensus {sto_input} {sto_output} 3 0.97 0.9 0.75 4 0.97 0.9 0.75 0.5 0.1"
    print(cmd)
    res = subprocess.run(cmd, shell = True, capture_output=True)
    print(res.stdout.decode())
    print(res.stderr.decode())
    return  

def draw_with_R2R(sto_input_weighted, pdf_output):
    ""
    cmd = f"r2r --disable-usage-warning {sto_input_weighted} {pdf_output}"
    print(cmd)
    res = subprocess.run(cmd, shell = True, capture_output=True)
    print(res.stdout.decode())
    print(res.stderr.decode())
    return 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sto_input", "-i", type=str, required=True)
    # parser.add_argument("--sto_output", "-isto", type=str, required=True)
    parser.add_argument("--weight", type=str, required=False)
    parser.add_argument("--pdf_output", "-o", type=str, required=True)
    args = parser.parse_args()

    if args.weight:
        sto_output = args.sto_input.replace(".sto", ".weighted.sto")
        append_custom_weight_to_stofile(args.sto_input, sto_output, args.weight)
    else:
        sto_output = args.sto_input.replace(".sto", ".gsc.sto")
        append_GSC_weight_to_stofile(args.sto_input, sto_output)
    draw_with_R2R(sto_output, args.pdf_output)
    
