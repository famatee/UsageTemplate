import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import  GATConv,GCNConv,SAGEConv
import scipy.sparse as sp
import datetime
from torch.autograd import Variable
import math


#控制分割单位，减少碎片化内存空间，充分利用gpu内存
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

class myDKT(nn.Module):
    def __init__(self, n_exercise, exercise_embed_dim, hidden_dim, layer_num, params, e2eIndicesTensor) -> None:
        super().__init__()
        # 练习题数量
        self.n_exercise = n_exercise
        self.n_kc = params.n_knowledge_concept
        self.exercise_embed_dim = exercise_embed_dim
        self.params = params
        self.hidden_dim = hidden_dim
        
        self.e_embed_layer=nn.Embedding(self.n_exercise+1,self.exercise_embed_dim)
        self.kc_embed_layer=nn.Embedding(self.n_kc,self.exercise_embed_dim)
        # 预先定义好e2e关系可以用哪些图卷积方法
        self.e2e_gat=GATConv(self.exercise_embed_dim, self.exercise_embed_dim,concat=False,dropout=self.params.e2e_dropout)
        self.e2e_gcn=GCNConv(self.exercise_embed_dim, self.exercise_embed_dim)
        self.e2e_sage=SAGEConv(self.exercise_embed_dim, self.exercise_embed_dim)
        # e2e关系
        self.e2eIndicesTensor=torch.LongTensor(e2eIndicesTensor).to('cuda')     
    def forward(self,e_inputs,a_inputs):
        
        # 初始化节点表征，练习只初始化了编号1~self.n_exercise的，0号是padding
        exercise_embed = self.e_embed_layer(torch.arange(1,self.n_exercise+1).to('cuda') ).to('cuda')
        # 把0号的padding给一个全0的向量拼最前面
        exercise_node_embedding=exercise_embed
        exercise_node_embedding=torch.cat((torch.zeros(1,exercise_node_embedding.shape[1]).to('cuda'),exercise_node_embedding),0)
        # 批量更新，只取这个batchsize涉及到的相关练习节点的e2e关系，如果关系不多也可以全图更新
        related_index=[]
        # related_index我是根据练习id和self.e2eIndicesTensor获取
        batch_e2eIndicesTensor=self.e2eIndicesTensor[:,related_index]
        # 传入GCN
        att_exercise_node_embedding=self.e2e_gcn(exercise_node_embedding,batch_e2eIndicesTensor)
        # att_exercise_node_embedding=self.e2e_gat(exercise_node_embedding,batch_e2eIndicesTensor)
        # att_exercise_node_embedding=self.e2e_sage(exercise_node_embedding,batch_e2eIndicesTensor)        
        
        
        
