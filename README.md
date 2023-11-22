# rfamgen
We tested all the following commands in macOS(v12.1) and Linux(v3.10) environment.  
## install with conda

```
conda env create -f requirements_rfamgen.yaml
conda activate rfamgen
```
RfamGen works with python3.7 and pytorch 1.8, but probaboly works with newer versions. Most processes are using the following packages. Please download one-by-one if necessary. 
```
pytorch
numpy
pandas
biopython
sklearn
nltk
infernal
viennarna
h5py
```
All the following commands are examples using RF00234. Computationa times by Intel(R) Core(TM) i5-7360U CPU @ 2.30GHz are commented.

## prep data
Data prep is perfomed based on a cmfile. A cmfile is a file format used in [infernal software](http://eddylab.org/infernal/). Here, we used a cmfile downloaded from Rfam DB. If you want to build, you can generated with `cmbuild` command of infernal. `cmbuild` requires `SS_cons`(consensus secondary structure) in an alignment file of stockholm format. `SS_cons` can be added by manually or consensus structure estimation such as [R-scape](http://eddylab.org/R-scape/).  
```
# Optional. For cleaning sequences in Rfam DB.

python scripts/get_tidy_sequences_from_fasta_rfam.py \
--seed_file ./datasets/Rfam.seed \
--rfam RF00234 \
--output_dir datasets/RF00234 \
--cpu 1 
```

<details><summary>help of `scripts/make_onehot_from_traceback.py` </summary><div>

```
usage: make_onehot_from_traceback.py [-h] [--fasta FASTA]
                                     [--traceback TRACEBACK] --cmfile CMFILE
                                     [--cpu CPU]

optional arguments:
  -h, --help            show this help message and exit
  --fasta FASTA         path to fasta file. Fasta file is automatically
                        aligned to cmfile and its traceback will be converted
                        to onehot.
  --traceback TRACEBACK
                        path to gzipped tracebackfile
  --cmfile CMFILE       path to cm file
  --cpu CPU             CPU cores for cmalign program. (default: 4)
```

</div></details>


```
python scripts/make_onehot_from_traceback.py \
--fasta datasets/RF00234/RF00234_unique_seed_removed.fa \
--cmfile ./datasets/RF00234/RF00234.cm \
--cpu 1
# real	3m24.649s
# user	3m0.349s
# sys	0m3.531s
```

After conversion to onehot expressions, split to train/valid/test.
```
python scripts/split_onehot_train_valid_test.py \
-i datasets/RF00234/RF00234_unique_seed_removed_notrunc_traceback_onehot_cm.h5 \
--train_ratio 0.7 --random_state 42
```

## generate weight
You can generate a weight file from the training data prepared above [[Marks, 2011](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0028766); [Morcos, 2011](https://www-pnas-org.kyoto-u.idm.oclc.org/doi/full/10.1073/pnas.1111471108)]. Typically, threshold is 0.01~0.2.  
The computation time highly depends on the data size.
```
python scripts/generate_weight.py \
--mode cm \
-i datasets/RF00234/RF00234_unique_seed_removed_notrunc_traceback_onehot_cm_train.h5 \
--threshold 0.1
# real    1m57.974s
# user    1m7.283s
# sys     1m15.031s
```

## train
`train.py` trains a VAE model,  generates config file of training, and optionally generates log/checkpoint files.  

<details><summary>
help of `scripts/train.py`
</summary><div>

```
usage: train.py [-h] --data_dir DATA_DIR --X_train X_TRAIN --w_train W_TRAIN
                --X_valid X_VALID --w_valid W_VALID [--suffix SUFFIX]
                [--hidden HIDDEN] [--z_dim Z_DIM] [--stride STRIDE]
                [--ker1 KER1] [--ch1 CH1] [--ker2 KER2] [--ch2 CH2]
                [--ker3 KER3] [--ch3 CH3] [--beta BETA] [--use_anneal]
                [--anneal_saturate_rate ANNEAL_SATURATE_RATE]
                [--anneal_rate ANNEAL_RATE] [--batch_size BATCH_SIZE]
                [--epoch EPOCH] [-lr LEARNING_RATE] [--clip CLIP]
                [--only_training] [--use_early_stopping]
                [--tolerance TOLERANCE] [--save_ckpt] [--ckpt_iter CKPT_ITER]
                [--random_seed RANDOM_SEED] [--log] [--log_dir LOG_DIR]
                [--print_every PRINT_EVERY]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   directory containing X_train etc.
  --X_train X_TRAIN     file name of training data
  --w_train W_TRAIN     file name of weight for training data
  --X_valid X_VALID     file name of validation data
  --w_valid W_VALID     file name of weight for validation data
  --suffix SUFFIX       suffix of output file name
  --hidden HIDDEN       dimension of hidden space flanking latent space
                        (default: 128)
  --z_dim Z_DIM         dimension of latent space (default: 16)
  --stride STRIDE       stride parameter of all convs (default: 1)
  --ker1 KER1           kernel size parameter of 1st conv (default: 5)
  --ch1 CH1             channel num parameter of 1st conv (default: 5)
  --ker2 KER2           kernel parameter of 2nd conv (default: 5)
  --ch2 CH2             channel num parameter of 2nd conv (default: 5)
  --ker3 KER3           kernel parameter of 3rd conv (default: 7)
  --ch3 CH3             channel num parameter of 3rd conv (default: 8)
  --beta BETA           maximum weight of KL divergence
  --use_anneal          flag for cyclic annealing for KL divergence
  --anneal_saturate_rate ANNEAL_SATURATE_RATE
                        maximum rate for cyclic annealing (default: 0.4)
  --anneal_rate ANNEAL_RATE
                        number of iteration for one increment of annealing
                        (default: 1)
  --batch_size BATCH_SIZE
                        batch size (defailt: 8)
  --epoch EPOCH         maximum epochs (default: 200)
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate (default: 1e-3)
  --clip CLIP           clipping value (default: 20)
  --only_training       only training
  --use_early_stopping  flag for early stopping
  --tolerance TOLERANCE
                        tolerance for early stopping (default: 3)
  --save_ckpt           flag for checkpoint saving
  --ckpt_iter CKPT_ITER
                        save checkpoint every this iteration (default: 3)
  --random_seed RANDOM_SEED
  --log                 flag for log
  --log_dir LOG_DIR     directory for log output
  --print_every PRINT_EVERY
                        iteration num to print log of learning (default: 20)
```
</div></details>

Example:
```
python scripts/train.py \
--data_dir datasets/RF00234 \
--X_train RF00234_unique_seed_removed_notrunc_traceback_onehot_cm_train.h5 \
--w_train RF00234_unique_seed_removed_notrunc_traceback_onehot_cm_train_weight_threshold0p1.h5 \
--X_valid RF00234_unique_seed_removed_notrunc_traceback_onehot_cm_valid.h5 \
--w_valid RF00234_unique_seed_removed_notrunc_traceback_onehot_cm_valid_weight_threshold0p1.h5 \
--epoch 3 \
--beta 1e-3 --use_anneal --use_early_stopping \
--log --log_dir ./outputs/RF00234

# real    0m16.900s
# user    0m12.500s
# sys     0m3.155s
```

## generation by sampling in the latent space.
`scripts/sampling_from_gauss.py` generates sequences from normal distribution in latent space. You need to input model params, config, and cmfile.  
Generation runs very fast unless the cmfile is very long. 
```
python scripts/sampling_from_gauss.py \
--config ./outputs/RF00234/config.yaml \
--ckpt ./outputs/RF00234/model_epoch3.pt \
--cmfile ./datasets/RF00234/RF00234.cm \
--outfasta ./outputs/RF00234/sampled_5seq.fa \
--n_samples 5

# real    0m7.447s
# user    0m6.201s
# sys     0m0.635s
```

# Bayesian estimation of cleavage kinetics
## install with conda 
```
conda env create -f requirements_bayesiankin.yaml
```
see more details in `/notebooks/bayesiankinetics/kinetics_analysis_bypyro.ipynb`  
It takes less than 1 day to estimate the kinetics of 2000 sequences x2 replicates with SVI.
The NGS data are available on [PRJNA1044007 of SRA](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1044007).