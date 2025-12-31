import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from torch.nn import Linear
from networks.Layers import Bottleneck_Construct, SeriesDecomposition, SeasonTrendFusion, MSFusion, PositionalEncoding
from networks.WITRAN import WITRAN_2DPSGMU_Encoder

class MSHTrans(nn.Module):
    def __init__(self, args, device):
        super(MSHTrans, self).__init__()
        self.n_feats = args["n_feats"]
        self.num_heads = args["head_num"]
        self.window_size = args["window_size"]
        self.max_seq_length = self.window_size
        self.scale_num = args["scale_num"] 
        self.lr = args["lr"]
        self.model_root = args["model_root"]
        self.device = device
        self.kernel_size  = args["pool_size_list"]
        self.conv_layers = Bottleneck_Construct(self.n_feats, args["pool_size_list"], self.n_feats)
        self.seq_length = [self.max_seq_length]
        self.num_layers = 1
        self.add_list = [20,15,10]
        self.window_size_list = [100,50,25]
        self.w_list = [10,5,5]
        self.h_list = [10,10,5]
        for i in range(self.scale_num - 1):
            self.seq_length.append(self.seq_length[-1] // args["pool_size_list"][i])

        self.pos_encoder = PositionalEncoding(self.n_feats * 2, 0.1, self.window_size)
            
        self.getfeature_list = nn.ModuleList()
        self.fc_list = nn.ModuleList()
        for i in range(self.scale_num):
            self.getfeature_list.append(WITRAN_2DPSGMU_Encoder(input_size=self.n_feats * 2, hidden_size=self.n_feats * 2, num_layers=self.num_layers, dropout=0., water_rows=self.w_list[i], water_cols=self.h_list[i]))
            self.fc_list.append(nn.Linear(self.num_layers * self.add_list[i] * self.n_feats * 2, self.window_size_list[i] * self.n_feats * 2))
        
        self.getfeature_d1 = WITRAN_2DPSGMU_Encoder(input_size=self.n_feats * 2, hidden_size=self.n_feats * 2, num_layers=self.num_layers, dropout=0., water_rows=10, water_cols=10)
        self.getfeature_proj_d1 = nn.Linear(self.n_feats * 2, self.n_feats)
        self.fc_d1 = nn.Linear(self.num_layers * 20 * self.n_feats * 2, self.window_size * self.n_feats * 2)

        self.series_decomposition = nn.ModuleList()
        self.season_trend_fusion = nn.ModuleList()
        for i in range(self.scale_num):
            self.series_decomposition.append(SeriesDecomposition(self.n_feats * 2))
            self.season_trend_fusion.append(SeasonTrendFusion(self.n_feats * 2, self.n_feats * 2))
            
        self.msfusion = MSFusion(self.n_feats * 2, args["pool_size_list"], self.n_feats * 2, self.seq_length)
        
        self.series_decomposition_d1 = SeriesDecomposition(self.n_feats)
        self.series_decomposition_d2 = SeriesDecomposition(args["n_feats"])
        self.season_trend_fusion_d1 = SeasonTrendFusion(self.n_feats, self.n_feats)
        self.sigmoid = nn.Sigmoid()
        
        self.node_embedding_proj = Linear(args["d_model"], self.n_feats)
        
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, 0.9)    
    
    
    def extract_downsample(self, x):
        window_x_list = [x]
        for i in range(self.scale_num - 1):
            idx = [j * pow(self.kernel_size[i], i + 1) for j in range(self.seq_length[i + 1])]
            sequence = x[:, idx, :]
            window_x_list.append(sequence)

        return window_x_list
    
    def extract_feature(self, x, isEncoder=True, i=0):
        B,_,D = x.size()

        if isEncoder:
            x = x.reshape(B,self.w_list[i],self.h_list[i],D)
            _, enc_hid_row, enc_hid_col = self.getfeature_list[i](x, batch_size=B, input_size=D, flag=0)         
        else:
            x = x.reshape(B,self.w_list[0],self.h_list[0],D)
            _, enc_hid_row, enc_hid_col = self.getfeature_d1(x, batch_size=B, input_size=D, flag=0)          

        _,W,H,_ = x.size()

        hidden_all = torch.cat([enc_hid_row, enc_hid_col], dim=2)
        hidden_all = hidden_all.reshape(hidden_all.shape[0], -1)

        last_output = hidden_all
        if isEncoder:
            last_output = self.fc_list[i](hidden_all)
        else:
            last_output = self.fc_d1(hidden_all)

        last_output = last_output.reshape(last_output.shape[0], H*W, -1)

        return last_output
        
    def encoder(self, x):
        window_ori_x = self.extract_downsample(x)    
        seq_enc = self.conv_layers(x) 
        for i in range(self.scale_num):
            seq_enc[i] = torch.concat([seq_enc[i], window_ori_x[i]], dim=-1)
            seq_enc[i] = self.pos_encoder(seq_enc[i])

        st_fusion_list = []
        for i in range(self.scale_num):      
            multi_head_node_emb = self.extract_feature(seq_enc[i], True, i)
            seasonality, trend = self.series_decomposition[i](multi_head_node_emb)
            st_fusion = self.season_trend_fusion[i](seq_enc[i], seasonality, trend) 
            st_fusion_list.append(st_fusion)
        fused_logits = self.msfusion(st_fusion_list)
        return fused_logits, st_fusion_list

    def decoder(self, x, fused_logits):  
        z_sea_1, z_trend_1 = self.series_decomposition_d1(x) 

        input = torch.concat([z_sea_1, fused_logits], dim=-1)
        input = self.pos_encoder(input)
        multi_head_node_emb = self.extract_feature(input, False)
        multi_head_node_emb = self.getfeature_proj_d1(multi_head_node_emb)

        z_sea_2, z_trend_2 = self.series_decomposition_d2(multi_head_node_emb)    
        z_trend_3 = z_trend_1 + z_trend_2

        results = self.season_trend_fusion_d1(x, z_sea_2, z_trend_3)    
        results = self.sigmoid(results)
  
        return results
        
    
    def forward(self, x):
        fused_logits, st_fusion_list = self.encoder(x)
        predict_logits = self.decoder(x, fused_logits)
 
        return predict_logits, st_fusion_list
    
    def con_constraint(self, st_fusion_list, temp=0.07):
        lcon_loss = 0.0
        batch_size = st_fusion_list[0].size(0)
        
        for i in range(len(st_fusion_list) - 1):
            z_s = st_fusion_list[i]
            z_s_plus = st_fusion_list[i+1]
            
            z_s_pooled = torch.mean(z_s, dim=1) 
            z_s_plus_pooled = torch.mean(z_s_plus, dim=1)
            
            z_s_norm = F.normalize(z_s_pooled, p=2, dim=1)
            z_s_plus_norm = F.normalize(z_s_plus_pooled, p=2, dim=1)
            
            logits = torch.mm(z_s_norm, z_s_plus_norm.t()) / temp
            
            labels = torch.arange(batch_size).to(self.device)
            lcon_loss += F.cross_entropy(logits, labels)
            
        return lcon_loss / (len(st_fusion_list) - 1)
        
    def train(self, args, dataloader):
        mse_func = nn.MSELoss(reduction="none")
        
        for epoch in range(1, args["nb_epoch"] + 1):
            logging.info("Training epoch: {}".format(epoch))

            loss_all_batch = 0.0
            
            for d in dataloader:
                ori_window_data = d[0].to(self.device)                            
                z, st_fusion_list = self(ori_window_data)            
                tgt = ori_window_data

                rec_loss = mse_func(z, tgt)
                rec_loss = torch.mean(rec_loss)
                con_loss = self.con_constraint(st_fusion_list)             
                loss_sum = rec_loss + con_loss

                loss_all_batch += loss_sum
                self.optimizer.zero_grad()
                loss_sum.backward()
                self.optimizer.step()

            self.scheduler.step()
            logging.info("Epoch: {} finished, loss is {:.4f}".format(epoch, loss_all_batch.cpu().detach().numpy()))    
            
    
    def predict_prob(self, dataloader):
        mse_func = nn.MSELoss(reduction="none")
        
        
        with torch.no_grad():
            loss_steps = []
            loss_all_var_steps = []
            z_steps = []
            
            for d in dataloader:
                ori_window_data =d[0].to(self.device)
                
                z, _ = self(ori_window_data)
                tgt = ori_window_data
                loss = mse_func(z, tgt) 
                
                loss_all_var = loss
                z_all_var = z
                
                loss = torch.mean(loss, dim=-1) 

                loss = loss[:, -1]

                loss_steps.append(loss.detach().cpu().numpy())
                loss_all_var_steps.append(loss_all_var.detach().cpu().numpy())
                z_steps.append(z_all_var.detach().cpu().numpy())
            anomaly_score = np.concatenate(loss_steps)
            loss_all_var_steps = np.concatenate(loss_all_var_steps, axis=0)
            z_steps = np.concatenate(z_steps, axis=0)
        
        return anomaly_score, loss_all_var_steps, z_steps
                    