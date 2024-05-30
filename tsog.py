import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from arguments import get_args

def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x ** 2).sum(1, keepdim=True))
    return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # self.W = nn.Parameter(torch.empty(size=(25, in_features, out_features)))
        self.W = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        # self.a = nn.Parameter(torch.empty(size=(25, 2*out_features, 1)))
        self.a1 = nn.Linear(out_features, 1, bias=False)
        self.a2 = nn.Linear(out_features, 1, bias=False)
        nn.init.xavier_uniform_(self.a1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a2.weight, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        Wh = self.W(h) # h.shape: (bz, N, in_features), Wh.shape: (bz, N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        # zero_vec = -9e15*torch.ones_like(e)
        # attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(e, dim=2)
        # attention = attention[:, :, -1]
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime, attention

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = self.a1(Wh)
        Wh2 = self.a2(Wh)
        # broadcast add
        e = Wh1 + Wh2.transpose(-2,-1)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, out_feature, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attn = None

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, out_feature, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x, self.attn = self.out_att(x)
        return x
    
class TimeEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=100):
        super(TimeEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)  # 预先计算好长度为 max_len 的位置编码，后面截取使用
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).unsqueeze(0)        # [1, 1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :, x.size(2)].requires_grad_(False)
        return self.dropout(x)

class TargetGraphEncoderLayer(nn.Module):
    def __init__(self, args):
        super(TargetGraphEncoderLayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 256))
        self.time_encoder = TimeEncoding(d_model=256, dropout=0.1)
        self.graph_proj = nn.Sequential(
            nn.Linear(256, 512), 
            nn.ReLU(), 
            nn.Linear(512, 256), 
            nn.ReLU()
        )
        self.target_proj = nn.Sequential(
            nn.Linear(256, 512), 
            nn.ReLU(), 
            nn.Linear(512, 256), 
            nn.ReLU()
        )
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=args.dropout_rate,
            batch_first=True
        )
        self.attn = None

    def forward(self, target, gat_embedding_memory):
        graph_key = gat_embedding_memory.permute(0, 2, 1, 3)    # [25, 16, 25, 256]
        graph_key = self.graph_proj(graph_key)   # [25, 16, 25, 256]
        graph_key_te = self.time_encoder(graph_key)       # [25, 16, 25, 256]
        target_query = target
        q = self.target_proj(target_query)      # [25, 1, 256]
        k = self.pool(graph_key_te.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).squeeze(1)        # [25, 25, 256]
        v = k
        _, self.attn = self.multihead_attn(q, k, v)     # [25, 1, 25]
        output = torch.sum(self.attn.unsqueeze(1).permute(0, 3, 1, 2) * graph_key_te.permute(0, 2, 1, 3), dim=1)    # [25, 16, 256]

        return output
    
class TSOG(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        action_space = args.action_space
        self.lstm_hidden_dim = args.hidden_state_sz
        self.num_cate = args.num_category
        dropout = args.dropout_rate
        text_embedding_dim = 300       # clip: 1024, glove: 300
        detection_feature_dim = 1024
        detection_info_dim = 6      # bbox、score、label/indicator
        graph_node_dim = 64
        resnet_embedding_channel = 512
        resnet_nin_channel = 64
        graph_nheads = 8
        graph_out_feature = 256
        alpha = 0.2

        self.image_size = 300
        self.gat_memory_len = args.gat_memory_len

        self.objects_text_embedding = torch.FloatTensor(np.load("./object_embeddings/glove6B_textfeatures_15cls.npy")).to(args.device)

        self.object_node_embedding = nn.Sequential(
            nn.Linear(text_embedding_dim + detection_feature_dim + detection_info_dim, 512), 
            nn.ReLU(), 
            nn.Linear(512, graph_node_dim),
        )

        self.nin_resnet = nn.Conv2d(resnet_embedding_channel, resnet_nin_channel, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    
        self.gat = GAT(graph_node_dim, graph_out_feature // graph_nheads, graph_out_feature, dropout, alpha, graph_nheads)
        # self.graph_pool = nn.AdaptiveAvgPool1d((49))
        self.graph_embedding = nn.Sequential(
            nn.Linear(graph_out_feature, 128),
            nn.ReLU(),
            nn.Linear(128, 49),
        )

        self.text_embedding_linear = nn.Linear(text_embedding_dim, graph_out_feature)
        self.temporal_graph_encoder = TargetGraphEncoderLayer(args=args)

        self.action_embedding = nn.Linear(action_space, 10)

        self.dropout = nn.Dropout(p=dropout)

        pointwise_in_channels = resnet_nin_channel + self.num_cate + 1 + 10
        # pointwise_in_channels = resnet_nin_channel + self.num_cate + 10
        self.pointwise = nn.Conv2d(pointwise_in_channels, 64, 1, 1)

        self.lstm_input_sz = 7 * 7 * 64
        self.lstm = nn.LSTM(self.lstm_input_sz, self.lstm_hidden_dim, 2, batch_first=True)
        self.critic_linear_1 = nn.Linear(self.lstm_hidden_dim, 64)
        self.critic_linear_2 = nn.Linear(64, 1)
        self.actor_linear = nn.Linear(self.lstm_hidden_dim, action_space)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.nin_resnet.weight.data.mul_(relu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)

        self.critic_linear_1.weight.data = norm_col_init(
            self.critic_linear_1.weight.data, 1.0
        )
        self.critic_linear_1.bias.data.fill_(0)
        self.critic_linear_2.weight.data = norm_col_init(
            self.critic_linear_2.weight.data, 1.0
        )
        self.critic_linear_2.bias.data.fill_(0)

        self.lstm.bias_ih_l0.data.fill_(0)
        self.lstm.bias_ih_l1.data.fill_(0)
        self.lstm.bias_hh_l0.data.fill_(0)
        self.lstm.bias_hh_l1.data.fill_(0)
    
    def embedding(self, state, target_class_embedding, action_probs, target_object, gat_embedding_memory):
        action_embedding = F.relu(self.action_embedding(action_probs)) 
        action_reshaped = action_embedding.view(-1, 10, 1, 1).repeat(1, 1, 7, 7)

        image_embedding = F.relu(self.nin_resnet(state))
        x = self.dropout(image_embedding)       # (25, 64, 7, 7)

        target_indicator = torch.zeros(target_class_embedding.shape[0], target_class_embedding.shape[1], 1).to(target_object.device)
        target_indicator.scatter_(1, target_object.unsqueeze(-1).unsqueeze(-1), 1)

        detection_targets = torch.cat((target_class_embedding, target_indicator), dim=-1)   # (25, 15, 1030)

        object_nodes = torch.cat((self.objects_text_embedding.unsqueeze(0).expand(detection_targets.shape[0], -1, -1), detection_targets), dim=2)    # (25, 15, 300+1030)

        object_nodes_embedding = self.object_node_embedding(object_nodes)   # (25, 15, 64)
        observation_node_embedding = self.avgpool(image_embedding).squeeze(dim=2).transpose(1,2)    # (25, 1, 64)

        observation_objects_graph = torch.cat((object_nodes_embedding, observation_node_embedding), dim=1)  # (25, 16, 64)

        gat_embedding = self.gat(observation_objects_graph).unsqueeze(1)     # (25, 1, 16, 256)
        gat_embedding_memory = torch.concat([gat_embedding, gat_embedding_memory[:, :-1]], dim=1)   # (25, 25, 16, 256)
        target_object_embedding = self.text_embedding_linear(self.objects_text_embedding[target_object])    # (25, 256)
        tog_embedding = self.temporal_graph_encoder(
            target_object_embedding.unsqueeze(1),   # (25, 1, 256) 
            gat_embedding_memory
        )       # (25, 16, 256)
        tog_embedding = self.dropout(self.graph_embedding(tog_embedding))     # (25, 16, 49)
        tog_embedding = tog_embedding.reshape(-1, self.num_cate+1, 7, 7)        # (25, 16, 7, 7)

        x = torch.cat((x, tog_embedding, action_reshaped), dim=1)       # (25, 64+16+10, 7, 7)
        x = F.relu(self.pointwise(x))           # (25, 64, 7, 7)
        x = self.dropout(x)
        out = x.view(x.size(0), -1)     # (25, 7*7*64)

        return out, gat_embedding_memory

    def a3clstm(self, embedding, prev_hx, prev_cx):
        embedding = embedding.reshape([-1, 1, self.lstm_input_sz])      #25*1*(64*7*7)
        output, (hx, cx) = self.lstm(embedding, (prev_hx.contiguous(), prev_cx.contiguous()))  
        
        x = output.reshape([-1, self.lstm_hidden_dim])    #25*512

        actor_out = self.actor_linear(x)   #512 - 4
        critic_out = self.critic_linear_1(x)   #512-64 
        critic_out = self.critic_linear_2(critic_out)   #64-1

        # (25, 4), (25, 1), (2, 25, 512), (2, 25, 512)
        return actor_out, critic_out, (hx, cx)

    def forward(self, state, target_object, hidden, action_probs, target_class_embedding, gat_embedding_memory):
        (hx, cx) = hidden               # 隐藏层特征

        x, gat_embedding_memory = self.embedding(state, target_class_embedding, action_probs, target_object, gat_embedding_memory)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, hx, cx)
        
        return (critic_out, actor_out, (hx, cx), gat_embedding_memory)
    
if __name__ == "__main__":
    args = get_args()
    model = TSOG(args)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print('Number of params:', n_parameters)
