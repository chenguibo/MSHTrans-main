import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.fft import series_decomp_multi
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[:, pos:pos+x.size(1), :]
        return self.dropout(x)

class DownConvLayer(nn.Module):
    def __init__(self, c_in, kernel_size):
        super(DownConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=kernel_size,
                                  stride=kernel_size)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class UpConvLayer(nn.Module):
    def __init__(self, c_in, kernel_size):
        super(UpConvLayer, self).__init__()
        self.upConv = nn.ConvTranspose1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=kernel_size,
                                  stride=kernel_size)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.upConv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class Bottleneck_Construct(nn.Module):
    def __init__(self, d_model, kernel_size, d_inner):
        super(Bottleneck_Construct, self).__init__()
        if not isinstance(kernel_size, list):
            self.conv_layers = nn.ModuleList([
                DownConvLayer(d_inner, kernel_size),
                DownConvLayer(d_inner, kernel_size),
                DownConvLayer(d_inner, kernel_size)
                ])
        else:
            self.conv_layers = []
            for i in range(len(kernel_size)):
                self.conv_layers.append(DownConvLayer(d_inner, kernel_size[i]))
            self.conv_layers = nn.ModuleList(self.conv_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):

        temp_input = enc_input.permute(0, 2, 1)
        all_inputs = []
        all_inputs.append(temp_input.permute(0, 2, 1))
        for i in range(len(self.conv_layers)):
            temp_input = self.conv_layers[i](temp_input)
            all_inputs.append(temp_input.permute(0, 2, 1))
        return all_inputs

class SeriesDecomposition(nn.Module):
    def __init__(self, d_model):
        super(SeriesDecomposition, self).__init__()
        self.trend_model = series_decomp_multi(kernel_size=[8, 16, 24]) 
        self.norm = nn.LayerNorm(d_model)
    def forward(self, time_series): 
        _, trend = self.trend_model(time_series)
        res = self.norm(time_series - trend)   
        return res, trend

class SeasonTrendFusion(nn.Module):
    def __init__(self, d_model, out_d):
        super(SeasonTrendFusion, self).__init__()
        self.x_weight = nn.Parameter(torch.ones(1) * 0.33)
        self.sea_weight = nn.Parameter(torch.ones(1) * 0.33)
        self.trend_weight = nn.Parameter(torch.ones(1) * 0.33)   
        self.norm = nn.LayerNorm(d_model)  
        self.d_ff = d_model * 2 
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_ff, out_d)
        )

    def forward(self, x, x_sea, x_trend):
        fused = self.x_weight * x + self.sea_weight * x_sea + self.trend_weight * x_trend
        fused = self.norm(fused)
        return self.feedforward(fused)

class MSFusion(nn.Module):
    def __init__(self, d_model, kernel_size, d_inner, seq_length):
        super(MSFusion, self).__init__()
        self.scale_num = len(kernel_size) + 1
        self.seq_length = seq_length
        self.w = nn.Parameter(torch.ones(self.scale_num, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4
        self.up = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)        
        self.conv_layers = nn.ModuleList([
            UpConvLayer(d_inner, k) for k in kernel_size
        ])
        
        self.d_ff = 128
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_ff, d_model // 2)
        )

    def forward(self, input_list):
        weight = torch.relu(self.w)
        weight = weight / (torch.sum(weight, dim=0) + self.epsilon)
        up_logit = input_list[-1]
        
        for i in range(len(input_list) - 1, 0, -1):
            logit = F.gelu(self.up(up_logit)) 
            temp_input = logit.permute(0, 2, 1)
            cur_up_logit = self.conv_layers[i - 1](temp_input).permute(0, 2, 1)        
            padding_num = self.seq_length[i - 1] - cur_up_logit.size(1)
            if padding_num > 0:
                cur_up_logit = F.pad(cur_up_logit, (0, 0, 0, padding_num), 'constant', 0)
            elif padding_num < 0:
                cur_up_logit = cur_up_logit[:, :self.seq_length[i-1], :]

            up_logit = weight[i] * cur_up_logit + weight[i-1] * input_list[i-1]
            up_logit = self.norm(up_logit)
            
        fused_logits = self.feedforward(up_logit)
        return fused_logits
