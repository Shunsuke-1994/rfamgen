import torch
import torch.nn as nn
import h5py
import util 

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

    
class View(nn.Module):
    def __init__(self, dim1, dim2):
        super(View, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, self.dim1, self.dim2)


class Encoder(nn.Module):
    """
    Convolutional encoder for C/G-VAE.
    """
    def __init__(self,
        seq_len,
        n_channel,
        hidden_size=435,
        z_dim=46,
        stride = 1,
        bn = True,
        add_fc = 0,
        conv_params = {"ker1":5, "ch1":5, "ker2":5, "ch2":5, "ker3":7, "ch3":8}
        ):
        """
        shape: (n_seq, n_rules)
        """
        super(Encoder, self).__init__()
        self.ker1, self.ch1 = conv_params["ker1"], conv_params["ch1"]
        self.ker2, self.ch2 = conv_params["ker2"], conv_params["ch2"]
        self.ker3, self.ch3 = conv_params["ker3"], conv_params["ch3"]
        self.bn    = bn
        self.stride = stride
        self.add_fc = add_fc
        self.seq_len = seq_len
        self.n_channel = n_channel

        # calculation of unit number after 3 convs.
        # see: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        def get_lastlen(len_sequence, dilation = 1, padding = 0):
            len_sequence = int(((len_sequence + 2*(self.ker1//2) - dilation * (self.ker1 - 1) - 1)/self.stride)+1)
            len_sequence = int(((len_sequence + 2*padding - dilation * (self.ker2 - 1) - 1)/1)+1)
            len_sequence = int(((len_sequence + 2*padding - dilation * (self.ker3 - 1) - 1)/1)+1)
            return len_sequence

        self.mu     = nn.Linear(hidden_size, z_dim)
        self.logvar = nn.Linear(hidden_size, z_dim)
        
        self.seq_encode = nn.Sequential(
            nn.Conv1d(self.n_channel, self.ch1, kernel_size=self.ker1, padding = self.ker1//2, stride=self.stride),
            nn.BatchNorm1d(self.ch1),
            nn.ReLU(),
            nn.Conv1d(self.ch1, self.ch2, kernel_size=self.ker2, padding = 0, stride=1),
            nn.BatchNorm1d(self.ch2),
            nn.ReLU(),
            nn.Conv1d(self.ch2, self.ch3, kernel_size=self.ker3, padding = 0, stride=1),
            nn.BatchNorm1d(self.ch3),
            nn.ReLU(),
            Flatten()
            )
        
        self.fcn = nn.Sequential(
            nn.Linear(get_lastlen(seq_len)*self.ch3, hidden_size),
            nn.ReLU()
            )
        
    def forward(self, x):
        """Encode x into a mean and variance of a Normal"""
        h = self.seq_encode(x)
        h = self.fcn(h) # (some, hidden_dim) x (batch, some) -> (batch, hidden_dim)
        return self.mu(h), self.logvar(h)


class CovarianceModelEncoder(nn.Module):
    """
    Convolutional encoder for CM-VAE.
    Applies a series of one-dimensional convolutions to a batch
    of tr/s/p encodings of the sequence of rules that generate
    an artithmetic expression.
    """
    def __init__(self,
        tr_len,
        s_len,
        p_len,
        hidden_size=435,
        z_dim=46,
        stride= 1,
        conv_params = {
            "ker1":5, "ch1":5,
            "ker2":5, "ch2":5,
            "ker3":7, "ch3":8
            }
        ):
        """
        shape: (n_seq, n_rules)
        """
        super(CovarianceModelEncoder, self).__init__()

        self.ker1, self.ch1 = conv_params["ker1"], conv_params["ch1"]
        self.ker2, self.ch2 = conv_params["ker2"], conv_params["ch2"]
        self.ker3, self.ch3 = conv_params["ker3"], conv_params["ch3"]
        
        self.bn    = True
        self.stride = stride
        self.n_fc = 0

        # calculation of unit number after 3 convs.
        # see: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        def get_lastlen(len_sequence, dilation = 1, padding = 0):
            len_sequence = int(((len_sequence + 2*(self.ker1//2) - dilation * (self.ker1 - 1) - 1)/self.stride)+1)
            len_sequence = int(((len_sequence + 2*padding - dilation * (self.ker2 - 1) - 1)/1)+1)
            len_sequence = int(((len_sequence + 2*padding - dilation * (self.ker3 - 1) - 1)/1)+1)
            return len_sequence

        self.mu     = nn.Linear(hidden_size, z_dim)
        self.logvar = nn.Linear(hidden_size, z_dim)
        
        self.tr_encode = nn.Sequential(
            nn.Conv1d(56, self.ch1, kernel_size=self.ker1, padding = self.ker1//2, stride=self.stride),
            nn.BatchNorm1d(self.ch1),
            nn.ReLU(),
            nn.Conv1d(self.ch1, self.ch2, kernel_size=self.ker2, padding = 0, stride=1),
            nn.BatchNorm1d(self.ch2),
            nn.ReLU(),
            nn.Conv1d(self.ch2, self.ch3, kernel_size=self.ker3, padding = 0, stride=1),
            nn.BatchNorm1d(self.ch3),
            nn.ReLU(),
            Flatten()
            )

        self.s_encode = nn.Sequential(
            nn.Conv1d(4, self.ch1, kernel_size=self.ker1, padding = self.ker1//2, stride=self.stride),
            nn.BatchNorm1d(self.ch1),
            nn.ReLU(),
            nn.Conv1d(self.ch1, self.ch2, kernel_size=self.ker2, padding = 0, stride=1),
            nn.BatchNorm1d(self.ch2),
            nn.ReLU(),
            nn.Conv1d(self.ch2, self.ch3, kernel_size=self.ker3, padding = 0, stride=1),
            nn.BatchNorm1d(self.ch3),
            nn.ReLU(),
            Flatten()
            )

        self.p_encode = nn.Sequential(
            nn.Conv1d(16, self.ch1, kernel_size=self.ker1, padding = self.ker1//2, stride=self.stride),
            nn.BatchNorm1d(self.ch1),
            nn.ReLU(),
            nn.Conv1d(self.ch1, self.ch2, kernel_size=self.ker2, padding = 0, stride=1),
            nn.BatchNorm1d(self.ch2),
            nn.ReLU(),
            nn.Conv1d(self.ch2, self.ch3, kernel_size=self.ker3, padding = 0, stride=1),
            nn.BatchNorm1d(self.ch3),
            nn.ReLU(),
            Flatten()
            )
        
        self.fcn = nn.Sequential(
            nn.Linear((get_lastlen(tr_len) + get_lastlen(s_len) + get_lastlen(p_len))*self.ch3, hidden_size),
            nn.ReLU()
            )
        
    def forward(self, x):
        """Encode x into a mean and variance of a Normal"""
        tr, s, p = x
        h_tr = self.tr_encode(tr)
        h_s  = self.s_encode(s)
        h_p  = self.p_encode(p)
        h    = torch.cat((h_tr, h_s, h_p), dim= 1)
        h    = self.fcn(h) # (some, hidden_dim) x (batch, some) -> (batch, hidden_dim)
        return self.mu(h), self.logvar(h)