

if __name__ == '__main__':
    import argparse
    import os 
    import sys 
    sys.path.append("src")
    import preprocess


    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', default = "", help='path to gzipped tracebackfile')
    parser.add_argument('--cmfile', default = "/Users/sumishunsuke/Desktop/RNA/genzyme/datasets/legacy/RF00234/RF00234.cm", help='path to cm file')
    parser.add_argument('--cpu', default=4, type = int)
    args = parser.parse_args()

    if args.fasta != "":
        preprocess.cmalign(
            cmfile = args.cmfile,
            seqfile = args.fasta,
            log = True,
            trunc = False, 
            suffix = "_notrunc",
            cpu = args.cpu) # --notruncation for dataset preprocessing.
        basename, _  = os.path.splitext(args.fasta)
        path_to_traceback = basename + "_notrunc_traceback.txt.gz"
    else:
        path_to_traceback = args.traceback
