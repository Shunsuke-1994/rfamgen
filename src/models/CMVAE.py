import h5py 
import numpy as np
import torch
from torch import nn
import util
import preprocess
from torch.utils.data import Dataset
from torch.distributions import Normal
from encoder import CovarianceModelEncoder
from decoder import CovarianceModelDecoder

class MyDataset(Dataset):
    def __init__(self, path, weight_path = ""):
        super().__init__()
        self.path = path
        self.data = h5py.File(path, "r")
        self.tr = self.data["tr"]
        self.s = self.data["s"]
        self.p = self.data["p"]
        if weight_path != "":
            self.weight = h5py.File(weight_path, "r")["weight"][:]
            self.weight = self.weight.reshape(self.weight.size, 1)
        else:
            self.weight = np.ones(self.tr.shape[0])
            self.weight = self.weight.reshape(self.weight.size, 1)
                
    def __len__(self):
        return self.tr.shape[0]
    
    def __getitem__(self, index):
        tr_tensor = torch.from_numpy(self.tr[index]).nan_to_num(0).transpose(-2, -1).float()
        s_tensor = torch.from_numpy(self.s[index]).nan_to_num(0).transpose(-2, -1).float()
        p_tensor = torch.from_numpy(self.p[index]).nan_to_num(0).transpose(-2, -1).float()
        w_tensor = torch.from_numpy(self.weight[index]).float()
        return tr_tensor, s_tensor, p_tensor, w_tensor


class CovarianceModelVAE(nn.Module):
    """
    CM-VAE. Encode and Decode CM or alignment on CM. 
    """
    def __init__(self,
        hidden_encoder_size,
        z_dim,
        hidden_decoder_size,
        tr_len,
        s_len,
        p_len,
        stride=1,
        conv_params = {
            "ker1":5, "ch1":5,
            "ker2":5, "ch2":5,
            "ker3":7, "ch3":8
            }
        ):
        super(CovarianceModelVAE, self).__init__()
        self.tr_len = tr_len
        self.s_len = s_len
        self.p_len = p_len
        self.z_dim = z_dim
        self.hidden_encoder_size = hidden_encoder_size
        self.hidden_decoder_size = hidden_decoder_size
        self.stride = stride
        self.conv_params = conv_params

        self.device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.encoder = CovarianceModelEncoder(
            self.tr_len, self.s_len, self.p_len,
            self.hidden_encoder_size,
            self.z_dim,
            stride = self.stride,
            conv_params = conv_params
            ).to(self.device)
        self.decoder = CovarianceModelDecoder(
            self.tr_len, self.s_len, self.p_len,
            self.hidden_decoder_size,
            self.z_dim,
            stride = self.stride,
            conv_params = conv_params
            ).to(self.device)

    def sample(self, mu, logvar):
        """Reparametrized sample from a N(mu, sigma) distribution
        input: (mu, logvar)
        """
        sigma  = (0.5*logvar).exp()
        normal = Normal(torch.zeros(mu.shape), torch.ones(sigma.shape))
        eps = normal.sample().to(self.device)
        z   = mu + eps*sigma
        return z

    def kl(self, mu, logvar):
        """KL divergence between two normal distributions"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z          = self.sample(mu, logvar)
        logits     = self.decoder(z)
        return logits

    def generate(self, z, sample=False):
        """
        Generate a valid expression from z using the decoder
        input must be a single latent variable. (z_dim vector)
        """
        if z.dim() == 1: 
            z = z.view(1, z.shape[0]) # increase the dim by 1. # render to tensro 1x z_dim to input to decoder.
        logits = self.decoder(z).squeeze().detach().numpy()
        return preprocess.onehot_decode(logits, sample = sample, vocab_type="rna")

    def load_model_from_ckpt(self, path_to_ckpt):
        self.load_state_dict(torch.load(path_to_ckpt, map_location=torch.device(self.device)))
        self.eval()
        return 

    @staticmethod
    def build_from_config(path):
        config         = util.load_config(path)
        HIDDEN_ENCODER = config["HIDDEN"]
        HIDDEN_DECODER = config["HIDDEN"]
        Z_DIM          = config["Z_DIM"]
        TR_LEN         = config["TR_WIDE"]
        S_LEN          = config["S_WIDE"]
        P_LEN          = config["P_WIDE"]
        STRIDE         = config["STRIDE"]
        if "KER1" in config:
            CONV_PARAMS = {
                "ker1":config["KER1"], "ch1":config["CH1"],
                "ker2":config["KER2"], "ch2":config["CH2"],
                "ker3":config["KER3"], "ch3":config["CH3"]
            }
        else: 
            CONV_PARAMS = {
                "ker1":5, "ch1":5,
                "ker2":5, "ch2":5,
                "ker3":7, "ch3":8
                }

        model = CovarianceModelVAE(
            hidden_encoder_size = HIDDEN_ENCODER,
            z_dim = Z_DIM,
            hidden_decoder_size = HIDDEN_DECODER,
            tr_len = TR_LEN, s_len = S_LEN, p_len = P_LEN,
            stride = STRIDE,
            conv_params= CONV_PARAMS
            )
        model.eval()
        return model