"""
This is the code for global-local transformer for brain age estimation

@email: heshengxgd@gmail.com

"""

import torch
import torch.nn as nn

import copy
import math

import numpy as np
import torch.nn.functional as F


# https://github.com/torcheeg/torcheeg/blob/main/torcheeg/models/gnn/dgcnn.py
class GraphConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool=True):

        super(GraphConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        nn.init.xavier_normal_(self.weight)
        #self.bias = None
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out


class Linear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def normalize_A(A: torch.Tensor, symmetry: bool=True, attn_weight=None) -> torch.Tensor:
    #A = F.relu(A)
    if symmetry:
 
        d = torch.sum(A, 2)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
        Lnorm = L-torch.eye(A.shape[1]).to(A.device)

    else:
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
        L = (L + attn_weight)/2
        Lnorm = L-torch.eye(68).reshape(1,68,68).repeat(attn_weight.size(0),1,1).to(A.device)
    return Lnorm


def generate_cheby_adj(A: torch.Tensor, num_layers: int) -> torch.Tensor:
    support = []
    for i in range(num_layers):
        if i == 0:
            temp = torch.eye(A.shape[1]).to(A.device)
            temp = temp.reshape(1,A.shape[1],A.shape[1])
            temp = temp.repeat(A.shape[0], 1, 1)
            support.append(temp)

        elif i == 1:
            support.append(A)
        else:
            temp = torch.matmul(2*A,support[-1],)-support[-2]
            support.append(temp)
    return support


class Chebynet(nn.Module):
    def __init__(self, in_channels: int, num_layers: int, out_channels: int):
        super(Chebynet, self).__init__()
        self.num_layers = num_layers
        self.gc1 = nn.ModuleList()
        self.leakrelu = nn.LeakyReLU()
        for i in range(num_layers):
            self.gc1.append(GraphConvolution(in_channels, out_channels))

    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        adj = generate_cheby_adj(L, self.num_layers)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result, adj[len(self.gc1)-1]
    
class Spa_Spe_se(nn.Module):
    def __init__(self, num_electrodes):
        super(Spa_Spe_se, self).__init__()

        self.channel_se1 = nn.Linear(num_electrodes,8, bias=False)
        self.channel_se2 = nn.Linear(8,num_electrodes, bias=False)
        self.fre_se1 = nn.Linear(5, 2, bias=False)
        self.fre_se2 = nn.Linear(2,5, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        B, time, ch, freq = x.size()

        x_c = x.permute(0,2,1,3).reshape(B,ch,time*freq) # (B, ch, Time,Fre,).
        x_f = x.permute(0,3,1,2).reshape(B,freq,time*ch) #(B, Fre, Time,ch)

        x_c = x_c.mean(dim=-1)
        x_f = x_f.mean(dim=-1)

        
        
        x_c = F.relu(self.channel_se1(x_c))
        x_c = self.sigmoid(self.channel_se2(x_c))
        x_c = x_c.view(B,1,ch,1)
        x_c = x*x_c.expand_as(x)

        x_f = F.relu(self.fre_se1(x_f))
        x_f = self.sigmoid(self.fre_se2(x_f))
        x_f = x_f.view(B,1,1,freq)
        x_f = x*x_f.expand_as(x)
     

        x = torch.cat((x_c,x_f), dim=-1) # B, ch, freq*2
        x = x.permute(0,2,1,3).reshape(B,ch,time*freq*2)

        return x






class DGCNN_local(nn.Module):
    r'''
    Dynamical Graph Convolutional Neural Networks (DGCNN). For more details, please refer to the following information.

    - Paper: Song T, Zheng W, Song P, et al. EEG emotion recognition using dynamical graph convolutional neural networks[J]. IEEE Transactions on Affective Computing, 2018, 11(3): 532-541.
    - URL: https://ieeexplore.ieee.org/abstract/document/8320798
    - Related Project: https://github.com/xueyunlong12589/DGCNN


    '''
    def __init__(self,
                 in_channels: int = 100*3,
                 num_electrodes: int = 114,
                 num_layers: int = 2,
                 hid_channels: int = 100*3,):
        super(DGCNN_local, self).__init__()
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.hid_channels = hid_channels
        self.num_layers = num_layers

        self.layer1 = Chebynet(in_channels, num_layers, hid_channels)
        self.BN1 = nn.BatchNorm1d(150)

        self.attn_softmax = nn.Softmax(dim=-1)
        self.edge_fc1 = nn.Linear(13,1,bias=False)
        self.edge_fc2 = nn.Linear(24,1,bias=False)
        self.edge_fc3 = nn.Linear(15,1,bias=False)
        self.edge_fc4 = nn.Linear(5,1,bias=False)
        self.edge_fc5 = nn.Linear(12,1,bias=False)
        self.edge_fc6 = nn.Linear(14,1,bias=False)
        self.edge_fc7 = nn.Linear(17,1,bias=False)
        self.edge_fc8 = nn.Linear(14,1,bias=False)

        self.edge_fc_list = [self.edge_fc1,self.edge_fc2,self.edge_fc3,self.edge_fc4,self.edge_fc5,self.edge_fc6,self.edge_fc7,self.edge_fc8]
        
        self.se_1 = Spa_Spe_se(13)
        self.se_2 = Spa_Spe_se(24)
        self.se_3 = Spa_Spe_se(15)
        self.se_4 = Spa_Spe_se(5)
        self.se_5 = Spa_Spe_se(12)
        self.se_6 = Spa_Spe_se(14)
        self.se_7 = Spa_Spe_se(17)
        self.se_8 = Spa_Spe_se(14)
        self.se_list = [self.se_1,self.se_2,self.se_3,self.se_4,self.se_5,self.se_6,self.se_7,self.se_8]
        


    def reset_parameters(self, pcc):
        self.pcc_edge = nn.Parameter(pcc)


        

    def forward(self, x: torch.Tensor, xcon) -> torch.Tensor:
        r'''

        '''

        B, time, ch, freq = x.size()
        x = x.permute(0,2,1,3).reshape(B,ch,time*freq)
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        x = x.reshape(B,ch,time,freq).permute(0,2,1,3)


        edge_x = xcon.cpu().detach().numpy()
        x_se_list = []
        ch_list = [13, 24, 15, 5, 12, 14, 17, 14]
        ch_len = [0,13, 13+24, 13+24+15, 13+24+15+5, 13+24+15+5+12, 13+24+15+5+12+14, 13+24+15+5+12+14+17, 114]
        edge_pcc = torch.zeros(B, 114,114).to(x.device)
        for i in range(8):
            local_ch = edge_x[:,ch_len[i]:ch_len[i+1],ch_len[i]:ch_len[i+1]]
            abs_pcc = []
            for r1 in range(ch_list[i]):
                for r2 in range(ch_list[i]):
                    if r1==r2:
                        abs_pcc.append(np.ones((local_ch.shape[0],local_ch.shape[1])))
                    else:
                        abs_pcc.append(abs(local_ch[:,r1] - local_ch[:,r2]))
            abs_pcc = np.array(abs_pcc)
            
            abs_pcc = abs_pcc.swapaxes(0,1)
            b,r,f = abs_pcc.shape
            abs_pcc = abs_pcc.reshape(b,ch_list[i],ch_list[i],f)
            abs_pcc = torch.FloatTensor(abs_pcc).to(x.device)
            ch_edge_pcc = F.relu(self.edge_fc_list[i](abs_pcc)).squeeze(dim=-1)
            ch_edge_pcc = self.attn_softmax(ch_edge_pcc)
            edge_pcc[:,ch_len[i]:ch_len[i+1],ch_len[i]:ch_len[i+1]] = ch_edge_pcc

            x_se = self.se_list[i](x[:,:,ch_len[i]:ch_len[i+1]])
            x_se_list.append(x_se)

        x = torch.cat(x_se_list, dim=1)
    
        L = normalize_A(edge_pcc,True )
       

        result, adj = self.layer1(x, L) # b, chan, fre*2

        return result
    

class DGCNN_global(nn.Module):
    r'''
    Dynamical Graph Convolutional Neural Networks (DGCNN). For more details, please refer to the following information.

    - Paper: Song T, Zheng W, Song P, et al. EEG emotion recognition using dynamical graph convolutional neural networks[J]. IEEE Transactions on Affective Computing, 2018, 11(3): 532-541.
    - URL: https://ieeexplore.ieee.org/abstract/document/8320798
    - Related Project: https://github.com/xueyunlong12589/DGCNN

    '''
    def __init__(self,
                 in_channels: int = 100*3,
                 num_electrodes: int = 114,
                 num_layers: int = 2,
                 hid_channels: int = 100*3,
                 num_classes: int = 3):
        super(DGCNN_global, self).__init__()
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.hid_channels = hid_channels
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.layer1 = Chebynet(in_channels, num_layers, hid_channels)
        self.BN1 = nn.BatchNorm1d(150)

        self.attn_softmax = nn.Softmax(dim=-1)
        self.edge_fc1 = nn.Linear(num_electrodes,1,bias=False)

        self.sigmoid = nn.Sigmoid()
        self.channel_se1 = nn.Linear(num_electrodes,8, bias=False)
        self.channel_se2 = nn.Linear(8,num_electrodes, bias=False)
        self.fre_se1 = nn.Linear(5, 2, bias=False)
        self.fre_se2 = nn.Linear(2,5, bias=False)


    def reset_parameters(self, pcc):
        self.pcc_edge = nn.Parameter(pcc)


        

    def forward(self, x: torch.Tensor, xcon) -> torch.Tensor:
        r'''
      
        '''

        B, time, ch, freq = x.size()

        x = x.permute(0,2,1,3).reshape(B,ch,time*freq)
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        x = x.reshape(B,ch,time,freq).permute(0,2,1,3)
       

        edge_x = xcon.cpu().detach().numpy()
        abs_pcc = []
        for r1 in range(ch):
            for r2 in range(ch):
                if r1==r2:
                    abs_pcc.append(np.ones((edge_x.shape[0],edge_x.shape[1])))
                else:
                    abs_pcc.append(abs(edge_x[:,r1] - edge_x[:,r2]))
        abs_pcc = np.array(abs_pcc)
        
        abs_pcc = abs_pcc.swapaxes(0,1)
        b,r,f = abs_pcc.shape
        abs_pcc = abs_pcc.reshape(b,ch,ch,f)
        abs_pcc = torch.FloatTensor(abs_pcc).to(x.device)
        edge_pcc = F.relu(self.edge_fc1(abs_pcc)).squeeze(dim=-1)
    
        L = normalize_A(self.attn_softmax(edge_pcc),True )
       
        
        x_c = x.permute(0,2,1,3).reshape(B,ch,time*freq) # (B, ch, Time,Fre,).
        x_f = x.permute(0,3,1,2).reshape(B,freq,time*ch) #(B, Fre, Time,ch)

        x_c = x_c.mean(dim=-1)
        x_f = x_f.mean(dim=-1)

        x_c = F.relu(self.channel_se1(x_c))
        x_c = self.sigmoid(self.channel_se2(x_c))
        x_c = x_c.view(b,1,ch,1)
        x_c = x*x_c.expand_as(x)

        x_f = F.relu(self.fre_se1(x_f))
        x_f = self.sigmoid(self.fre_se2(x_f))
        x_f = x_f.view(b,1,1,freq)
        x_f = x*x_f.expand_as(x)

        x = torch.cat((x_c,x_f), dim=-1) # B, ch, freq*2
        x = x.permute(0,2,1,3).reshape(B,ch,time*freq*2)

        result, adj = self.layer1(x, L) # b, chan, fre*2
        
        return result
    
    
class GlobalAttention(nn.Module):
    def __init__(self, 
                 transformer_num_heads=2,
                 hidden_size=100,
                 transformer_dropout_rate=0.5):
        super().__init__()
        
        self.num_attention_heads = transformer_num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(transformer_dropout_rate)
        self.proj_dropout = nn.Dropout(transformer_dropout_rate)
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.attn_score = None
        
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # batch, num_head, rois, length//num_heads
    
    def forward(self,locx,glox):
        locx_query_mix = self.query(locx)
        glox_key_mix = self.key(glox)
        glox_value_mix = self.value(glox)
        
        query_layer = self.transpose_for_scores(locx_query_mix)
        key_layer = self.transpose_for_scores(glox_key_mix)
        value_layer = self.transpose_for_scores(glox_value_mix)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        
        
        attention_probs = self.attn_dropout(attention_probs)
        self.attn_score = attention_probs
        
        context_layer = torch.matmul(attention_probs, value_layer)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # batch, rois, num_heads, length//num_heads
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # batch, rois, length
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        
        return attention_output
    

class GlobalAttention_fusion(nn.Module):
    def __init__(self, 
                 transformer_num_heads=2,
                 hidden_size=100,
                 transformer_dropout_rate=0.5):
        super().__init__()
        
        self.num_attention_heads = transformer_num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(transformer_dropout_rate)
        self.proj_dropout = nn.Dropout(transformer_dropout_rate)
        
        self.softmax = nn.Softmax(dim=-1)

        self.attn_score = None
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # batch, num_head, electro, length
    
    def forward(self,locx):
        locx_query_mix = self.query(locx)
        glox_key_mix = self.key(locx)
        glox_value_mix = self.value(locx)
        
        query_layer = self.transpose_for_scores(locx_query_mix)
        key_layer = self.transpose_for_scores(glox_key_mix)
        value_layer = self.transpose_for_scores(glox_value_mix)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        

        attention_probs = self.attn_dropout(attention_probs)
        self.attn_score = attention_probs
        
        
        context_layer = torch.matmul(attention_probs, value_layer)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # batch, rois, num_heads, length//num_heads
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # batch, rois, length
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        
        return attention_output


class convBlock(nn.Module):
    def __init__(self,inplace,outplace,kernel_size=3,padding=1):
        super().__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(inplace,outplace,kernel_size=kernel_size,padding=padding,bias=False)
        self.bn1 = nn.BatchNorm1d(outplace)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x
    
class Feedforward(nn.Module):
    def __init__(self,inplace,outplace):
        super().__init__()
        
        self.conv1 = convBlock(inplace,outplace,kernel_size=1,padding=0)
        self.conv2 = convBlock(outplace,outplace,kernel_size=1,padding=0)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    

class community_classifier(nn.Module):
    def __init__(self,hidden_size):
        super(community_classifier, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size,9)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    


class BN_BrainTF(nn.Module):
    def __init__(self,num_rois=114,
                 nblock=4,
                 drop_rate=0.5,
                 hidden_size = 300,
                 num_classes=3):
        """
  
        """  
        super().__init__()    

        self.global_feat = DGCNN_global(num_electrodes=114)
        self.local_feat = DGCNN_local(num_electrodes=114)

        self.fusion_nblock = nblock
        self.local_nblock = 2
        self.global_nblock =2
        self.num_rois = num_rois
        self.hidden_size = hidden_size
        
        region_n = [13,24,15,5,12,14,17,14]
        self.attnlist_list = nn.ModuleList()
        self.fftlist_list = nn.ModuleList()
        for region in range(len(region_n)):
            self.fftlist__ = nn.ModuleList()
            self.attn__ = nn.ModuleList()
            for n in range(self.fusion_nblock):
                atten = GlobalAttention(
                    transformer_num_heads=4,
                    hidden_size=hidden_size,
                    transformer_dropout_rate=drop_rate)
                self.attn__.append(atten)
                fft = Feedforward(inplace=region_n[region]*2,
                                  outplace=region_n[region])
                self.fftlist__.append(fft)
            self.attnlist_list.append(self.attn__)
            self.fftlist_list.append(self.fftlist__)

        self.fusion_tf = GlobalAttention_fusion(transformer_num_heads=4,hidden_size=hidden_size,transformer_dropout_rate=drop_rate)
        self.fusion_ff = Feedforward(inplace=num_rois*2,outplace=num_rois)
        
        self.fusion_tf2 = GlobalAttention_fusion(transformer_num_heads=4,hidden_size=hidden_size,transformer_dropout_rate=drop_rate)
        self.fusion_ff2 = Feedforward(inplace=num_rois*2,outplace=num_rois)
   
            
        self.emotion_classifier = nn.Linear(hidden_size,num_classes)

        self.node_rearranged_len = [0,13, 13+24, 13+24+15, 13+24+15+5, 13+24+15+5+12, 13+24+15+5+12+14, 13+24+15+5+12+14+17, 114]

        self.local_dis = community_classifier(hidden_size=hidden_size)
        self.fc1 = nn.Linear(num_rois*hidden_size, hidden_size)
        
    def forward(self,xinput, xcon):

        B,_,_,_ = xinput.size()
        outlist = []
        diff_out = []
     
        xglo = self.global_feat(xinput, xcon)
        xglo_dis = self.local_dis(xglo.mean(dim=1))
        diff_out.append(xglo_dis)

        loc = self.local_feat(xinput, xcon)
        for y in range(len(self.node_rearranged_len)-1):
             
            xloc = loc[:,self.node_rearranged_len[y]:self.node_rearranged_len[y+1]]
            xloc_dis = self.local_dis(xloc.mean(dim=1))
            diff_out.append(xloc_dis)
                
            for n in range(self.fusion_nblock):

                tmp = self.attnlist_list[y][n](xloc,xglo) 
                tmp = torch.cat([tmp,xloc],1) # batch, rois*2, length
                tmp = self.fftlist_list[y][n](tmp)

                xloc = xloc + tmp
            
            outlist.append(xloc) 
        outlist = torch.cat(outlist, dim=1)

        temp_outlist = self.fusion_tf(outlist)
        temp_outlist = torch.cat([temp_outlist,outlist],1)
        temp_outlist = self.fusion_ff(temp_outlist)
        outlist = outlist + temp_outlist
        
        temp_outlist = self.fusion_tf2(outlist)
        temp_outlist = torch.cat([temp_outlist,outlist],1)
        temp_outlist = self.fusion_ff2(temp_outlist)
        outlist = outlist + temp_outlist
        outlist = torch.flatten(outlist,1)

        xloc_total = F.relu(self.fc1(outlist))
        local_out = self.emotion_classifier(xloc_total)
        local_out = F.log_softmax(local_out, dim=1)

        return local_out, diff_out

      
    

        