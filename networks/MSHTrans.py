import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from torch_geometric.data import data as D
from torch.nn import Linear
from networks.MAHLayer import Multi_Adaptive_Hypergraph
from networks.Layers import Bottleneck_Construct, SeriesDecomposition, SeasonTrendFusion, MSFusion, PositionalEncoding
from networks.HyperGraphConv import HypergraphConv
from networks.STAR import STAR

class MSHTrans(nn.Module):
    def __init__(self, args, device):
        super(MSHTrans, self).__init__()
        self.n_feats = args["n_feats"]
        self.window_size = args["window_size"]
        self.max_seq_length = self.window_size
        self.hyper_num = args["scale_num"] 
        self.lr = args["lr"]
        self.model_root = args["model_root"]
        self.device = device
        self.kernel_size  = args["pool_size_list"]
        self.conv_layers = Bottleneck_Construct(self.n_feats, args["pool_size_list"], self.n_feats)
        self.seq_length = [self.max_seq_length]
        for i in range(self.hyper_num - 1):
            self.seq_length.append(self.seq_length[-1] // args["pool_size_list"][i])
        self.hyper_head = args["head_num"]

        self.multi_adpive_hypergraph = Multi_Adaptive_Hypergraph(args, self.device)
        
        self.pos_encoder = PositionalEncoding(self.n_feats * 2, 0.1, self.window_size)
        
        self.hyconv_list = nn.ModuleList()
        for i in range (self.hyper_num):
            self.hyconv = nn.ModuleList()
            for j in range(self.hyper_head):
                self.hyconv.append(HypergraphConv(self.n_feats * 2, self.n_feats * 2))
            self.hyconv_list.append(self.hyconv)
            
        # STAR modules per scale (predefined and registered)
        self.star_list = nn.ModuleList()
        for i in range(self.hyper_num):
            L_i = self.seq_length[i]
            D_i = self.n_feats * 2
            self.star_list.append(STAR(d_series=L_i, d_core=D_i))
        
        # Separate STAR for decoder to avoid sharing encoder parameters
        self.star_decoder = STAR(d_series=self.seq_length[0], d_core=self.n_feats * 2)
        self.decoder_proj = nn.Linear(self.n_feats * 2, self.n_feats)

        self.series_decomposition = nn.ModuleList()
        self.season_trend_fusion = nn.ModuleList()
        for i in range(self.hyper_num):
            seq_len = self.seq_length[i]
            self.series_decomposition.append(SeriesDecomposition(seq_len, self.n_feats * 2))
            self.season_trend_fusion.append(SeasonTrendFusion(self.n_feats * 2, self.n_feats * 2))
            
        self.msfusion = MSFusion(self.n_feats * 2, args["pool_size_list"], self.n_feats * 2, self.seq_length)
        
        self.series_decomposition_d1 = SeriesDecomposition(self.max_seq_length, self.n_feats)
        self.series_decomposition_d2 = SeriesDecomposition(self.max_seq_length, args["n_feats"])
        self.hyconv_d1 = nn.ModuleList()
        for j in range(self.hyper_head):
            self.hyconv_d1.append(HypergraphConv(self.n_feats * 2, self.n_feats))
        self.season_trend_fusion_d1 = SeasonTrendFusion(self.n_feats, self.n_feats)
        self.sigmoid = nn.Sigmoid()
        
        self.node_embedding_proj = Linear(args["d_model"], self.n_feats)
        
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, 0.9)    
    
    
    def extract_downsample(self, x):
        window_x_list = [x]
        for i in range(self.hyper_num - 1):
            idx = [j * pow(self.kernel_size[i], i + 1) for j in range(self.seq_length[i + 1])]
            sequence = x[:, idx, :]
            window_x_list.append(sequence)

        return window_x_list
        
    def encoder(self, x, hyper_graph_indicies):
        # x: (B, L, D)

        # Multi-scale Window Generator
        window_ori_x = self.extract_downsample(x)    
        seq_enc = self.conv_layers(x) 
        for i in range(self.hyper_num):
            seq_enc[i] = torch.concat([seq_enc[i], window_ori_x[i]], dim=-1)
            seq_enc[i] = self.pos_encoder(seq_enc[i])
        #  seq_enc[i]: (B, L_i, D*2)

        st_fusion_list = []
        for i in range(self.hyper_num):
            # region 【编码器超图模块注释】
            # input: seq_enc[i] (batch_size, seq_len_i, n_feats*2)

            # hyperedge_indices = torch.tensor(hyper_graph_indicies[i]).to(self.device)    
            # node_value = seq_enc[i].permute(0,2,1).to(self.device)
            # edge_indices, node_indices = hyperedge_indices[1], hyperedge_indices[0]
            # num_edges = edge_indices.max().item() + 1
            # num_nodes = node_value.size(2)  

            # # Intra-scale Hypergraph Attention
            # indices = torch.stack([edge_indices, node_indices])  
            # values = torch.ones(edge_indices.size(0), device=node_value.device)
            # adj_matrix = torch.sparse_coo_tensor(indices, values, (num_edges, num_nodes), device=node_value.device)
            # node_value = node_value.permute(2, 0, 1).contiguous()
            # edge_features = torch.sparse.mm(adj_matrix, node_value.view(num_nodes, -1)).view(num_edges, node_value.size(1), node_value.size(2))
            
            # # Multi-head Hypergraph Convolution
            # multi_head_hyconv = self.hyconv_list[i]
            # output_list = []
            # for j in range(self.hyper_head):
            #     output = multi_head_hyconv[j](seq_enc[i], hyperedge_indices, edge_features).permute(1, 0, 2)
            #     output_list.append(output)
            # multi_head_node_emb = torch.mean(torch.stack(output_list, dim = -1), dim = -1)     
            # multi_head_node_emb = multi_head_node_emb + seq_enc[i]

            # output: multi_head_node_emb (batch_size, seq_len_i, n_feats*2)
            # endregion

            # STAR (use pre-created module; STAR expects input shape (B, D, L))
            # seq_enc[i]: (B, L, D) -> permute to (B, D, L)
            multi_head_node_emb = self.star_list[i](seq_enc[i].permute(0, 2, 1)).permute(0, 2, 1)

            # Series Decomposition  
            seasonality, trend = self.series_decomposition[i](multi_head_node_emb)
            
            # Seasonality and Trend Fusion
            st_fusion = self.season_trend_fusion[i](seq_enc[i], seasonality, trend) 
            st_fusion_list.append(st_fusion)
        # Multi-scale Fusion
        fused_logits = self.msfusion(st_fusion_list)
        return fused_logits, st_fusion_list

    def decoder(self, x, fused_logits, hyperedge_index):
        edge_features = {}   
        node_value = x.permute(0,2,1)
        hyperedge_index = hyperedge_index.to(self.device)
        edge_indices, node_indices = hyperedge_index[1], hyperedge_index[0]
        num_edges = edge_indices.max().item() + 1
        num_nodes = node_value.size(2) 
        indices = torch.stack([edge_indices, node_indices])  
        values = torch.ones(edge_indices.size(0), device=node_value.device)
        adj_matrix = torch.sparse_coo_tensor(indices, values, (num_edges, num_nodes), device=node_value.device)
        node_value = node_value.permute(2, 0, 1).contiguous()
        edge_features = torch.sparse.mm(adj_matrix, node_value.view(num_nodes, -1)).view(num_edges, node_value.size(1), node_value.size(2))
        
        # input: x: (B, L, D) 形状验证一下

        # Series Decomposition 1
        z_sea_1, z_trend_1 = self.series_decomposition_d1(x) 

        # input/output: z_sea_1、z_trend_1 (B, L, D)

        # region 【Multi-head Hypergraph Convolution】
        # input_hyconv = torch.concat([z_sea_1, fused_logits], dim=-1)
        # input_hyconv = self.pos_encoder(input_hyconv)
        # output_list = []
        # for i in range(self.hyper_head):
        #     output = self.hyconv_d1[i](input_hyconv, hyperedge_index, edge_features).permute(1, 0, 2) 
        #     output_list.append(output)
        # multi_head_node_emb = torch.mean(torch.stack(output_list, dim = -1), dim = -1)

        # endregion output: multi_head_node_emb (batch_size, seq_len, n_feats)

        # STAR (use pre-created module for full-resolution scale)
        input_star = torch.concat([z_sea_1, fused_logits], dim=-1)
        input_star = self.pos_encoder(input_star)
        # input_star: (B, L, D) -> permute to (B, D, L) for STAR, then back
        multi_head_node_emb = self.star_decoder(input_star.permute(0, 2, 1)).permute(0, 2, 1)
        # multi_head_node_emb = input_star
        if multi_head_node_emb.size(-1) != self.n_feats:
            multi_head_node_emb = self.decoder_proj(multi_head_node_emb)

        # Series Decomposition 2
        z_sea_2, z_trend_2 = self.series_decomposition_d2(multi_head_node_emb)    
        z_trend_3 = z_trend_1 + z_trend_2

        # Seasonality and Trend Fusion
        results = self.season_trend_fusion_d1(x, z_sea_2, z_trend_3)    
        results = self.sigmoid(results)
  
        return results
        
    
    def forward(self, x, hyper_graph_indicies, fused_hypergraph):
        fused_logits, st_fusion_list = self.encoder(x, hyper_graph_indicies)
        predict_logits = self.decoder(x, fused_logits, fused_hypergraph)
 
        return predict_logits, st_fusion_list
    
    def hyperedge_constraint(self, window_ori_x, hyper_graph_indicies, node_embedding_list, edge_embedding_list, edge_retain_list):

        loss_hyperedge_all_scale = 0.0
        loss_node_all_scale = 0.0
        for i in range(self.hyper_num):
            hyper_graph_index = torch.tensor(hyper_graph_indicies[i]).to(self.device)
            
            edge_embedding = edge_embedding_list[i][edge_retain_list[i], :]
            x = window_ori_x[i]
            node_value = x.permute(0,2,1)
            
            edge_indices, node_indices = hyper_graph_index[1], hyper_graph_index[0]

            num_edges = edge_indices.max().item() + 1
            num_nodes = node_value.size(2)  

            indices = torch.stack([edge_indices, node_indices])  
            values = torch.ones(edge_indices.size(0), device=node_value.device)
            adj_matrix = torch.sparse_coo_tensor(indices, values, (num_edges, num_nodes), device=node_value.device)

            node_value = node_value.permute(2, 0, 1).contiguous()
            edge_features = torch.sparse.mm(adj_matrix, node_value.view(num_nodes, -1)).view(num_edges, node_value.size(1), node_value.size(2))

            edge_features = torch.mean(edge_features, dim=1)

            loss_hyper = 0.0
            for k in range(edge_features.size(0)):
                for m in range(edge_features.size(0)):
                    inner_product = torch.sum(edge_features[k, :] * edge_features[m, :], dim=-1, keepdim=True)
                    norm_q_i = torch.norm(edge_features[k, :], dim=-1, keepdim=True)
                    norm_q_i = torch.clamp(norm_q_i, min=1e-4)
                    norm_q_j = torch.norm(edge_features[m, :], dim=-1, keepdim=True)
                    norm_q_j = torch.clamp(norm_q_j, min=1e-4)
                    alpha = inner_product / (norm_q_i * norm_q_j)


                    distan = torch.norm(edge_embedding[k, :] - edge_embedding[m, :], dim=0, keepdim=True) 

                    loss_item = alpha * distan + (1 - alpha) * (torch.clamp(torch.tensor(4.2) - distan, min=0.0))
                    loss_hyper = loss_hyper + torch.abs(torch.mean(loss_item))
                
            loss_hyper = loss_hyper / ((edge_features.size(0) + 1) ** 2)
            loss_hyperedge_all_scale = loss_hyperedge_all_scale + loss_hyper
        
            node_embedding = self.node_embedding_proj(node_embedding_list[i])
            x_i = torch.index_select(node_embedding, dim=0, index=hyper_graph_index[0])
            x_j = torch.index_select(edge_features, dim=0, index=hyper_graph_index[1])
            loss_node = abs(torch.mean(x_i - x_j))
            loss_node_all_scale = loss_node_all_scale + loss_node
            
        
        loss_hyperedge_all_scale = 0.1 * loss_hyperedge_all_scale
        
        return loss_hyperedge_all_scale, loss_node_all_scale
        
    def Laplacian_constraint(self, H, Z):
        A = H @ H.t()
        D = torch.diag(torch.sum(A, dim=1))
        L = D - A
        
        L_expanded = L.unsqueeze(0).expand(Z.size(0), -1, -1)
        
        LZ = torch.bmm(L_expanded, Z)
        
        Z_T_LZ = torch.bmm(Z.transpose(1, 2), LZ)
        
        loss = torch.einsum('bii->b', Z_T_LZ)
        loss = torch.mean(loss)
        return loss
    
    def con_constraint(self, st_fusion_list, temp=0.07):
        """
        跨尺度对比学习损失 (InfoNCE)
        st_fusion_list: 包含不同尺度特征的列表，每个形状为 (B, L_i, D)
        """
        lcon_loss = 0.0
        batch_size = st_fusion_list[0].size(0)
        
        for i in range(len(st_fusion_list) - 1):
            # 1. 提取相邻尺度的特征 Z(s) 和 Z(s+1)
            z_s = st_fusion_list[i]
            z_s_plus = st_fusion_list[i+1]
            
            # 2. 全局时间平均池化：将 (B, L, D) 转换为 (B, D)
            # 因为不同尺度长度 L 不同，通过池化将其对齐到相同维度
            z_s_pooled = torch.mean(z_s, dim=1) 
            z_s_plus_pooled = torch.mean(z_s_plus, dim=1)
            
            # 3. 特征归一化
            z_s_norm = F.normalize(z_s_pooled, p=2, dim=1)
            z_s_plus_norm = F.normalize(z_s_plus_pooled, p=2, dim=1)
            
            # 4. 计算相似度矩阵 (B, B)
            logits = torch.mm(z_s_norm, z_s_plus_norm.t()) / temp
            
            # 5. 正样本是对角线元素 (即同一窗口的不同尺度表示)
            labels = torch.arange(batch_size).to(self.device)
            lcon_loss += F.cross_entropy(logits, labels)
            
        return lcon_loss / (len(st_fusion_list) - 1)
        
    def train(self, args, dataloader):
        mse_func = nn.MSELoss(reduction="none")
        
        for epoch in range(1, args["nb_epoch"] + 1):
            logging.info("Training epoch: {}".format(epoch))

            loss_all_batch = 0.0

            hyper_graph_indicies, H_list, edge_retain_list, fused_hypergraph, fused_retain_edge = self.multi_adpive_hypergraph() 
            
            for d in dataloader:
                ori_window_data = d[0].to(self.device)
                              
                z, st_fusion_list = self(ori_window_data, hyper_graph_indicies, fused_hypergraph)

                
                tgt = ori_window_data

                rec_loss = mse_func(z, tgt)
                rec_loss = torch.mean(rec_loss)
                
                # region 【超图相关损失函数】
                # window_ori_x = self.extract_downsample(ori_window_data)
                # node_embedding_list, edge_embedding_list = self.multi_adpive_hypergraph.get_embeddings()
                # loss_hyperedge_all_scale, loss_node_all_scale = self.hyperedge_constraint(window_ori_x, hyper_graph_indicies, node_embedding_list, edge_embedding_list, edge_retain_list)
                
                # loss_laplacian = 0.0
                
                # for i in range(self.hyper_num):
                #     H = torch.mm(node_embedding_list[i], edge_embedding_list[i].t())
                #     H = F.softmax(F.relu(self.multi_adpive_hypergraph.alpha * H))
                #     loss_laplacian = loss_laplacian + self.Laplacian_constraint(H, window_ori_x[i])

                
                
                # loss_sum = rec_loss + loss_hyperedge_all_scale + loss_node_all_scale + loss_laplacian
                # endregion

                # 跨尺度对比学习损失
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
            hyper_graph_indicies, H_list, edge_retain_list, fused_hypergraph, fused_retain_edge = self.multi_adpive_hypergraph(train=False) 
            
            for d in dataloader:
                ori_window_data =d[0].to(self.device)
                
                z, _ = self(ori_window_data, hyper_graph_indicies, fused_hypergraph)
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
                    