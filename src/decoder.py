import torch
import torch.nn as nn
import torch.nn.functional as F

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


class Decoder(nn.Module):
    """
    Convolutional decoder for C/G-VAE. 
    """
    def __init__(self,
        seq_len,
        n_channel,
        hidden_size=435,
        z_dim=46,
        stride=1,
        bn = True,
        add_fc = 0,
        conv_params = {"ker1":5, "ch1":5, "ker2":5, "ch2":5, "ker3":7, "ch3":8}
        ):
        """
        shape: (n_seq, n_rules)
        """
        super(Decoder, self).__init__()

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
        def get_padding_param(seq_len):
            
            def outpads(leng):
                # encode側の計算
                conv1_in = leng
                conv2_in = int(((conv1_in + 2*(self.ker1//2) - self.ker1)/self.stride)+1) 
                conv3_in = int(((conv2_in + 2*0 - self.ker2)/1)+1)
                conv3_out= int(((conv3_in + 2*0 - self.ker3)/1)+1)

                # これがoutput_padding = 0の時, convtransposeで出てくる長さ
                deconv3_out = int((conv3_out   - 1) * 1 - 2*0 + (self.ker3 - 1) + 1)
                deconv2_out = int((deconv3_out - 1) * 1 - 2*0 + (self.ker2 - 1) + 1)
                deconv1_out = int((deconv2_out - 1) * self.stride - 2*(self.ker1//2) + (self.ker1 - 1) + 1)

                outpad3      = conv3_in - deconv3_out
                outpad2      = conv2_in - deconv2_out
                outpad1      = conv1_in - deconv1_out
                return conv3_out, (outpad1, outpad2, outpad3)
            
            seq_conv3out, seq_outpads = outpads(seq_len)
            return seq_conv3out, seq_outpads
   

        self.leng, seq_outpads = get_padding_param(self.seq_len)
        self.fcn_x2 = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.ch3*self.leng),
            View(self.ch3, self.leng),
            nn.BatchNorm1d(self.ch3),
            nn.ReLU()
            )

        outpad1, outpad2, outpad3 = seq_outpads
        self.seq_decode = nn.Sequential(
            nn.ConvTranspose1d(self.ch3, self.ch2, kernel_size=self.ker3, padding = 0, stride=1, output_padding=outpad3),
            nn.BatchNorm1d(self.ch2),
            nn.ReLU(),
            nn.ConvTranspose1d(self.ch2, self.ch1, kernel_size=self.ker2, padding = 0, stride=1, output_padding=outpad2),
            nn.BatchNorm1d(self.ch1),
            nn.ReLU(),
            nn.ConvTranspose1d(self.ch1, self.n_channel, kernel_size=self.ker1, padding = self.ker1//2, stride=self.stride, output_padding=outpad1),
            nn.BatchNorm1d(self.n_channel),
            nn.ReLU()
            )

    def forward(self, z):
        """Encode x into a mean and variance of a Normal"""
        h = self.fcn_x2(z)
        return self.seq_decode(h)


class CovarianceModelDecoder(nn.Module):
    """
    Convolutional encoder for CM-VAE(split type).
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
        stride = 1,
        conv_params = {"ker1":5, "ch1":5, "ker2":5, "ch2":5, "ker3":7, "ch3":8}
        ):
        """
        shape: (n_seq, n_rules)
        """
        super(CovarianceModelDecoder, self).__init__()

        self.ker1, self.ch1 = conv_params["ker1"], conv_params["ch1"]
        self.ker2, self.ch2 = conv_params["ker2"], conv_params["ch2"]
        self.ker3, self.ch3 = conv_params["ker3"], conv_params["ch3"]
        self.bn    = True
        self.stride = stride
        self.n_fc = 0
        
        # calculation of unit number after 3 convs.
        # see: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        def get_padding_param(tr_len, s_len, p_len):
            
            def outpads(leng):
                # calc on encoder side
                conv1_in = leng
                conv2_in = int(((conv1_in + 2*(self.ker1//2) - self.ker1)/self.stride)+1) 
                conv3_in = int(((conv2_in + 2*0 - self.ker2)/1)+1)
                conv3_out= int(((conv3_in + 2*0 - self.ker3)/1)+1)

                deconv3_out = int((conv3_out   - 1) * 1 - 2*0 + (self.ker3 - 1) + 1)
                deconv2_out = int((deconv3_out - 1) * 1 - 2*0 + (self.ker2 - 1) + 1)
                deconv1_out = int((deconv2_out - 1) * self.stride - 2*(self.ker1//2) + (self.ker1 - 1) + 1)

                outpad3      = conv3_in - deconv3_out
                outpad2      = conv2_in - deconv2_out
                outpad1      = conv1_in - deconv1_out
                return conv3_out, (outpad1, outpad2, outpad3)
            
            tr_conv3out, tr_outpads = outpads(tr_len)
            s_conv3out, s_outpads = outpads(s_len)
            p_conv3out, p_outpads = outpads(p_len)
            return (tr_conv3out, s_conv3out, p_conv3out), tr_outpads, s_outpads, p_outpads
   

        self.leng, tr_outpads, s_outpads, p_outpads = get_padding_param(tr_len, s_len, p_len)
        self.fcn_x2 = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.ch3*sum(self.leng)),
            View(self.ch3, sum(self.leng)),
            nn.BatchNorm1d(self.ch3),
            nn.ReLU()
            )

        outpad1, outpad2, outpad3 = tr_outpads
        self.tr_decode = nn.Sequential(
            nn.ConvTranspose1d(self.ch3, self.ch2, kernel_size=self.ker3, padding = 0, stride=1, output_padding=outpad3),
            nn.BatchNorm1d(self.ch2),
            nn.ReLU(),
            nn.ConvTranspose1d(self.ch2, self.ch1, kernel_size=self.ker2, padding = 0, stride=1, output_padding=outpad2),
            nn.BatchNorm1d(self.ch1),
            nn.ReLU(),
            nn.ConvTranspose1d(self.ch1, 56, kernel_size=self.ker1, padding = self.ker1//2, stride=self.stride, output_padding=outpad1),
            nn.BatchNorm1d(56),
            nn.ReLU()
            )
        
        outpad1, outpad2, outpad3 = s_outpads
        self.s_decode = nn.Sequential(
            nn.ConvTranspose1d(self.ch3, self.ch2, kernel_size=self.ker3, padding = 0, stride=1, output_padding=outpad3),
            nn.BatchNorm1d(self.ch2),
            nn.ReLU(),
            nn.ConvTranspose1d(self.ch2, self.ch1, kernel_size=self.ker2, padding = 0, stride=1, output_padding=outpad2),
            nn.BatchNorm1d(self.ch1),
            nn.ReLU(),
            nn.ConvTranspose1d(self.ch1, 4, kernel_size=self.ker1, padding = self.ker1//2, stride=self.stride, output_padding=outpad1),
            nn.BatchNorm1d(4),
            nn.ReLU()
            )
        
        outpad1, outpad2, outpad3 = p_outpads
        self.p_decode = nn.Sequential(
            nn.ConvTranspose1d(self.ch3, self.ch2, kernel_size=self.ker3, padding = 0, stride=1, output_padding=outpad3),
            nn.BatchNorm1d(self.ch2),
            nn.ReLU(),
            nn.ConvTranspose1d(self.ch2, self.ch1, kernel_size=self.ker2, padding = 0, stride=1, output_padding=outpad2),
            nn.BatchNorm1d(self.ch1),
            nn.ReLU(),
            nn.ConvTranspose1d(self.ch1, 16, kernel_size=self.ker1, padding = self.ker1//2, stride=self.stride, output_padding=outpad1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )

    def forward(self, z):
        """Encode x into a mean and variance of a Normal"""
        h = self.fcn_x2(z)
        h_tr, h_s, h_p = h[:, :, :self.leng[0]], h[:, :, self.leng[0]:-self.leng[2]], h[:, :, -self.leng[2]:]
        return self.tr_decode(h_tr), self.s_decode(h_s), self.p_decode(h_p)