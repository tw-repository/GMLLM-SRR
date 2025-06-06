
device = "cuda:0"
# from utils.transformer import Transformer
import torch
import os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from vgg_v1 import resnet50
# from torch_geometric.nn import GatedGraphConv, GCNConv
from einops import repeat, rearrange
from timm import create_model
from transformers import AutoTokenizer, AutoModel
from torchvision.models import resnet101
import warnings
import csv

warnings.filterwarnings("ignore", category=UserWarning, message=".*no max_length.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ViT_imagenet = create_model('vit_large_patch16_224', pretrained=False, num_classes=512)
ViT_imagenet_1 = create_model('vit_large_patch16_224', pretrained=False, num_classes=512)
ViT_imagenet_2 = create_model('vit_large_patch16_224', pretrained=False, num_classes=512)
# model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
print("---success load pretrain ViT---")
ViT_dict = ViT_imagenet.state_dict()
pretrained_model = torch.load(r'/your_path/jx_vit_large_p16_224-4ee7a4dc.pth')
del pretrained_model['head.weight']
del pretrained_model['head.bias']
ViT_dict.update(pretrained_model)
ViT_imagenet.load_state_dict(ViT_dict)
# 1 PRE
ViT_dict = ViT_imagenet_1.state_dict()
pretrained_model = torch.load(r'/your_path/jx_vit_large_p16_224-4ee7a4dc.pth')
del pretrained_model['head.weight']
del pretrained_model['head.bias']
ViT_dict.update(pretrained_model)
ViT_imagenet_1.load_state_dict(ViT_dict)
# 2 PRE
ViT_dict = ViT_imagenet_2.state_dict()
pretrained_model = torch.load(r'/your_path/jx_vit_large_p16_224-4ee7a4dc.pth')
del pretrained_model['head.weight']
del pretrained_model['head.bias']
ViT_dict.update(pretrained_model)
ViT_imagenet_2.load_state_dict(ViT_dict)

print("----------------------------pretrained vit loading over-----------------------")

model_path = '/your_path/bert-base-uncased'
# model_path = '/mnt/disk/tw_data/roberta-base'
# model_path = '/mnt/disk/tw_data/electra-base-discriminator'

tokenizer = AutoTokenizer.from_pretrained(model_path)
lan_model = AutoModel.from_pretrained(model_path)
print("----------------------pretrained language model loading over-----------------------")


class person_pair(nn.Module):
    def __init__(self, num_classes=6):
        super(person_pair, self).__init__()

        self.pair = ViT_imagenet_1
        self.person_a = ViT_imagenet
        self.person_b = self.person_a
        self.bboxes = nn.Linear(10, 512)

    # x1 = union, x2 = object1, x3 = object2, x4 = bbox geometric info
    def forward(self, x1, x2, x3, x4):
        pair = self.pair(x1)
        pa = self.person_a(x2)
        pb = self.person_b(x3)
        bbox = self.bboxes(x4)

        return bbox, pair, pa, pb


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: [N, in_features] where N is the number of nodes
        # adj: adjacency matrix of size [N, N]
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        return output + self.bias


class GCN(nn.Module):
    def __init__(self, num_node_features, hidden_size):
        super(GCN, self).__init__()
        self.conv1 = GraphConvLayer(num_node_features, hidden_size)
        self.conv2 = GraphConvLayer(hidden_size, num_node_features)

    def forward(self, x, adj):
        x = F.relu(self.conv1(x, adj))
        x = self.conv2(x, adj)
        return x


class CrossModalAttention(nn.Module):
    def __init__(self, text_dim, img_dim, num_heads, temperature):
        super(CrossModalAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = text_dim ** -0.5
        self.temperature = temperature

        self.text_qkv_proj = nn.Linear(text_dim, 3 * num_heads * (text_dim // num_heads))
        self.img_qkv_proj = nn.Linear(img_dim, 3 * num_heads * (img_dim // num_heads))

        self.modality_weights = nn.Parameter(torch.ones(num_heads, 2)).cuda()

    def forward(self, text_features, img_features, vis_class, lan_class):

        B = text_features.size(0)

        text_features = text_features[:, np.newaxis, :]
        img_features = img_features[:, np.newaxis, :]

        b, n, _, h = *text_features.shape, self.num_heads
        qkv_text = self.text_qkv_proj(text_features).chunk(3, dim=-1)
        text_q, text_k, text_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv_text)

        b, n, _, h = *img_features.shape, self.num_heads
        qkv_img = self.text_qkv_proj(img_features).chunk(3, dim=-1)
        img_q, img_k, img_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv_img)

        text_to_img_scores = torch.einsum('bhid,bhjd->bhij', text_q, img_k) * self.scale
        img_to_text_scores = torch.einsum('bhid,bhjd->bhij', img_q, text_k) * self.scale

        img_un = self.calculate_entropy(vis_class)
        text_un = self.calculate_entropy(lan_class)
        img_loss_weights, text_loss_weights = self.calculate_alpha(img_un, text_un)

        img_un_weights = img_loss_weights * torch.ones((B, self.num_heads)).to(img_features.device)
        text_un_weights = text_loss_weights * torch.ones((B, self.num_heads)).to(text_features.device)

        text_to_img_weights = self.adjust_attention_scores(text_to_img_scores, img_un_weights)
        img_to_text_weights = self.adjust_attention_scores(img_to_text_scores, text_un_weights)

        text_to_img_weighted_features = torch.einsum('bhij,bhjd->bhid', text_to_img_weights, img_v)
        img_to_text_weighted_features = torch.einsum('bhij,bhjd->bhid', img_to_text_weights, text_v)

        out_text_to_img_weighted_features = rearrange(text_to_img_weighted_features, 'b h n d -> b n (h d)')
        out_img_to_text_weighted_features = rearrange(img_to_text_weighted_features, 'b h n d -> b n (h d)')

        return out_text_to_img_weighted_features, out_img_to_text_weighted_features, \
               sum(text_to_img_scores).mean(), sum(text_to_img_weights).mean(), \
               sum(img_to_text_scores).mean(), sum(img_to_text_weights).mean()

    def calculate_entropy(self, tensor):

        tensor = F.softmax(tensor, dim=1)
        tensor = tensor.clamp(min=1e-12)

        entropy = -torch.sum(torch.mean(tensor * torch.log(tensor), dim=0), dim=0)
        return entropy

    def calculate_alpha(self, entropy_g, entropy_t):
        exp_neg_Ug = torch.exp(-entropy_g)
        exp_neg_Ut = torch.exp(-entropy_t)

        alpha_g = exp_neg_Ug / (exp_neg_Ug + exp_neg_Ut)
        alpha_t = exp_neg_Ut / (exp_neg_Ug + exp_neg_Ut)

        return alpha_g, alpha_t


    def adjust_attention_scores(self, attention_scores, loss_weights):

        loss_weights = loss_weights.view(-1, self.num_heads)
        loss_weights = torch.relu(torch.exp(-loss_weights / self.temperature))

        modality_weights = F.softmax(self.modality_weights, dim=-1)
        modality_weights = modality_weights[:, 0].unsqueeze(-1) * loss_weights.unsqueeze(-1)

        modality_weights = modality_weights.unsqueeze(-1)
        attention_weights = attention_scores * modality_weights

        return attention_weights


num_node_features = 512
hidden_size = 512
SIZE = 512
dim = 2048
heads = 8
dim_head = 64
dropout = 0.
depth = 2
mlp_dim = 1024


class network(nn.Module):
    def __init__(self, num_classes=6):
        super(network, self).__init__()

        self.person_pair = person_pair(num_classes)
        self.scene_feature = ViT_imagenet_2
        self.lan_feature = lan_model

        self.fc_language = nn.Linear(768, SIZE)

        self.fc_visual = nn.Linear(SIZE * 5, SIZE)
        self.visual_graph = GCN(num_node_features, hidden_size)
        self.fc_visual_fuse = nn.Linear(SIZE * 2, SIZE)

        self.fc_class_v = nn.Linear(SIZE, num_classes)
        self.fc_class_l = nn.Linear(SIZE, num_classes)
        self.fc_class_lvm = nn.Linear(SIZE * 4, num_classes)

        self.attention_model = CrossModalAttention(text_dim=512, img_dim=512,
                                                   num_heads=8, temperature=0.8).cuda()

        self.fc_class = nn.Linear(SIZE * 3, num_classes)

    def forward(self, union, b1, b2, b_geometric, full_im, img_rel_num, edge_index, lan):
        scene_feature = self.scene_feature(full_im)
        scene_new = scene_feature[0].repeat(img_rel_num[0], 1)
        for i, num in enumerate(img_rel_num[1:]):
            scene_new = torch.cat([scene_new, scene_feature[i + 1].repeat(num, 1)], dim=0)
        bbox, pair, pa, pb = self.person_pair(union, b1, b2, b_geometric)

        language = []
        for i, num in enumerate(img_rel_num):
            language.extend([lan[i]] * num)

        sentence_features = []
        for text in language:
            lan_token = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=SIZE)
            lan_token = {k: v.cuda() for k, v in lan_token.items()}
            with torch.no_grad():
                outputs = self.lan_feature(**lan_token)
            last_hidden_states = outputs.last_hidden_state
            cls_index = lan_token['input_ids'].squeeze().tolist().index(tokenizer.cls_token_id)
            sentence_feature = last_hidden_states[:, cls_index, :].detach()
            sentence_features.append(sentence_feature)

        sentence_features_tensor = torch.stack(sentence_features).squeeze(dim=1)
        language_feature = self.fc_language(sentence_features_tensor)

        visual_feature_cat = torch.cat((bbox, pair, pa, pb, scene_new), 1)
        visual_feature_fuse = self.fc_visual(visual_feature_cat)

        max_node = max(edge_index[0].max(), edge_index[1].max())
        adj = torch.zeros((max_node + 1, max_node + 1), dtype=torch.float).cuda()
        for start_node, connected_nodes in enumerate(edge_index):
            for end_node in connected_nodes:
                if start_node != end_node:
                    adj[start_node, end_node] = 1

        current_dim = adj.shape[0]
        expected_dim = visual_feature_fuse.shape[0]

        dif = abs(current_dim - expected_dim)
        current_rows, current_cols = adj.shape
        if expected_dim > current_rows:
            padding_col = torch.zeros((current_rows, dif)).cuda()
            adj = torch.cat((adj, padding_col), dim=1)
            padding_row = torch.zeros((dif, current_cols + dif)).cuda()
            adj = torch.cat((adj, padding_row), dim=0)
        elif expected_dim < current_rows:
            adj = adj[:expected_dim, :expected_dim]
        visual_reason = self.visual_graph(visual_feature_fuse, adj)

        visual_reason_fuse = self.fc_visual_fuse(torch.cat((visual_reason, visual_feature_fuse), 1))
        vis_class = self.fc_class_v(visual_reason_fuse)
        lan_class = self.fc_class_l(language_feature)

        cross_img, cross_text, att1t, att2t, att1i, att2i\
            = self.attention_model(language_feature, visual_reason_fuse, vis_class, lan_class)

        fuse_class = self.fc_class_lvm(torch.cat((cross_img.squeeze(1), cross_text.squeeze(1),
                                                  language_feature, visual_reason_fuse), 1))

        return fuse_class
