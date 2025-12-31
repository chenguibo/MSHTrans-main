import torch
import torch.nn as nn
import math
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class WITRAN_2DPSGMU_Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, water_rows, water_cols, res_mode='none'):
        super(WITRAN_2DPSGMU_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.water_rows = water_rows
        self.water_cols = water_cols
        self.res_mode = res_mode
        self.W_first_layer = torch.nn.Parameter(torch.empty(6 * hidden_size, input_size + 2 * hidden_size))
        self.W_other_layer = torch.nn.Parameter(torch.empty(num_layers - 1, 6 * hidden_size, 4 * hidden_size))
        self.B = torch.nn.Parameter(torch.empty(num_layers, 6 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def linear(self, input, weight, bias, batch_size, slice, Water2sea_slice_num):
        a = F.linear(input, weight)
        if slice < Water2sea_slice_num:
            a[:batch_size * (slice + 1), :] = a[:batch_size * (slice + 1), :] + bias
        return a

    def forward(self, input, batch_size, input_size, flag):
        if flag == 1: 
            input = input.permute(2, 0, 1, 3)
        else: 
            input = input.permute(1, 0, 2, 3)
        Water2sea_slice_num, _, Original_slice_len, _ = input.shape
        Water2sea_slice_len = Water2sea_slice_num + Original_slice_len - 1 
        hidden_slice_row = torch.zeros(Water2sea_slice_num * batch_size, self.hidden_size).to(input.device)
        hidden_slice_col = torch.zeros(Water2sea_slice_num * batch_size, self.hidden_size).to(input.device)
        input_transfer = torch.zeros(Water2sea_slice_num, batch_size, Water2sea_slice_len, input_size).to(input.device)

        for r in range(Water2sea_slice_num):
            input_transfer[r, :, r:r+Original_slice_len, :] = input[r, :, :, :]

        hidden_row_all_list = []
        hidden_col_all_list = []
        for layer in range(self.num_layers):
            if layer == 0:
                a = input_transfer.reshape(Water2sea_slice_num * batch_size, Water2sea_slice_len, input_size)
                W = self.W_first_layer 
            else:
                a = F.dropout(output_all_slice, self.dropout, self.training)
                if layer == 1:
                    layer0_output = a
                W = self.W_other_layer[layer-1, :, :]
                hidden_slice_row = hidden_slice_row * 0
                hidden_slice_col = hidden_slice_col * 0
            B = self.B[layer, :] 

            output_all_slice_list = []
            for slice in range (Water2sea_slice_len):
                gate = self.linear(torch.cat([hidden_slice_row, hidden_slice_col, a[:, slice, :]],
                    dim = -1), W, B, batch_size, slice, Water2sea_slice_num)
                sigmod_gate, tanh_gate = torch.split(gate, 4 * self.hidden_size, dim = -1)
                sigmod_gate = torch.sigmoid(sigmod_gate)
                tanh_gate = torch.tanh(tanh_gate)
                update_gate_row, output_gate_row, update_gate_col, output_gate_col = sigmod_gate.chunk(4, dim = -1)
                input_gate_row, input_gate_col = tanh_gate.chunk(2, dim = -1) 
                hidden_slice_row = torch.tanh(
                    (1-update_gate_row)*hidden_slice_row + update_gate_row*input_gate_row) * output_gate_row
                hidden_slice_col = torch.tanh(
                    (1-update_gate_col)*hidden_slice_col + update_gate_col*input_gate_col) * output_gate_col
                output_slice = torch.cat([hidden_slice_row, hidden_slice_col], dim = -1) 
                output_all_slice_list.append(output_slice)

                if slice >= Original_slice_len - 1:
                    need_save_row_loc = slice - Original_slice_len + 1
                    hidden_row_all_list.append(
                        hidden_slice_row[need_save_row_loc*batch_size:(need_save_row_loc+1)*batch_size, :])
                if slice >= Water2sea_slice_num - 1:
                    hidden_col_all_list.append(
                        hidden_slice_col[(Water2sea_slice_num-1)*batch_size:, :])
                hidden_slice_col = torch.roll(hidden_slice_col, shifts=batch_size, dims = 0)

            if self.res_mode == 'layer_res' and layer >= 1:
                output_all_slice = torch.stack(output_all_slice_list, dim = 1) + layer0_output
            else:
                output_all_slice = torch.stack(output_all_slice_list, dim = 1)

        hidden_row_all = torch.stack(hidden_row_all_list, dim = 1)
        hidden_col_all = torch.stack(hidden_col_all_list, dim = 1)
        hidden_row_all = hidden_row_all.reshape(batch_size, self.num_layers, Water2sea_slice_num, hidden_row_all.shape[-1])
        hidden_col_all = hidden_col_all.reshape(batch_size, self.num_layers, Original_slice_len, hidden_col_all.shape[-1])
        if flag == 1:
            return output_all_slice, hidden_col_all, hidden_row_all
        else:
            return output_all_slice, hidden_row_all, hidden_col_all


if __name__ == '__main__':
    x1 = torch.randn(128, 20, 55).to(device)
    x1 = x1.reshape(128,4,5,55)
    B,W,H,C = x1.size()
    num_layers = 1

    Model = WITRAN_2DPSGMU_Encoder(input_size=C, hidden_size=C, num_layers=num_layers, dropout=0., water_rows=W, water_cols=H).to(device)
    output_all_slice, enc_hid_row, enc_hid_col = Model(x1, batch_size=B, input_size=C, flag=0)
    print(output_all_slice.shape,enc_hid_row.shape,enc_hid_col.shape)

    FC = nn.Linear(num_layers * (W + H) * C, W*H*C).to(device)
    hidden_all = torch.cat([enc_hid_row, enc_hid_col], dim=2)
    hidden_all = hidden_all.reshape(hidden_all.shape[0], -1)
    last_output = FC(hidden_all)
    last_output = last_output.reshape(last_output.shape[0], H*W, -1)
    print(last_output.shape)

