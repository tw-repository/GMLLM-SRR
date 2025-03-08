# test.py
"""
test the val acc
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import copy
import pickle as pkl
from PIL import Image
from dataset import DataSet
import random
import warnings
import csv

warnings.filterwarnings('ignore')

seed = 0
os.environ['PYTHONHASHSEED'] = str(seed) 
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed)  
torch.cuda.manual_seed_all(seed)  
random.seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description='PyTorch Relationship')
parser.add_argument('--mode', dest="mode", default="PIPA", type=str,
                    help='PISC_Fine, PISC_Coarse or PIPA')
parser.add_argument('--network', default='GGNN_attn', type=str, help='Network name.')
parser.add_argument('--scene', default=True, type=bool, help='whether scene node exists')
parser.add_argument('--model_file', default=r"/your_save_model.pth.tar",
                    type=str, help='Model file to be tested.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (defult: 4)')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 1)')
parser.add_argument('--scale-size', default=256, type=int, help='input size')
parser.add_argument('--crop-size', default=224, type=int, help='crop size')

args = parser.parse_args()
print(args)

if args.mode == "PISC_Fine":
    data_dir = r"/your_image_path"
    num_class = 6
    train_list = r"/data/PISCF_train.pkl"
    test_list = r"/data/PISCF_test.pkl"
elif args.mode == "PISC_Coarse":
    data_dir = r"/your_image_path"
    num_class = 3
    train_list = r"/data/PISCC_train.pkl"
    test_list = r"/data/PISCC_test.pkl"
else:
    data_dir = r"/your_image_path"
    num_class = 16
    train_list = r"/data/PIPA_train.pkl"
    test_list = r"/data/PIPA_test.pkl"


def vg_collate(data):
    Name = []
    Union = []
    Obj1 = []
    Obj2 = []
    Bpos = []
    Target = []
    Full_im = []
    Img_rel_num = []
    language = []

    for d in data:
        name, union, obj1, obj2, bpos, target, full_im, img_rel_num, lan = d
        # print(name)
        # print(union.shape)
        Name.append(name)
        Union.append(union)
        Obj1.append(obj1)
        Obj2.append(obj2)
        Bpos.append(bpos)
        Target.append(target)
        Full_im.append(full_im)
        Img_rel_num.append(img_rel_num)
        language.append(lan)

    Union = torch.cat(Union, 0)
    Obj1 = torch.cat(Obj1, 0)
    Obj2 = torch.cat(Obj2, 0)
    Bpos = torch.cat(Bpos, 0)
    Target = torch.cat(Target, 0)
    Full_im = torch.cat(Full_im, 0)
    Full_im = Full_im.view(-1, 3, 224, 224)
    Img_rel_num = torch.cat(Img_rel_num, 0)

    return Name, Union, Obj1, Obj2, Bpos, Target, Full_im, Img_rel_num, language


def get_test_set(data_dir, test_list):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    scale_size = args.scale_size
    crop_size = args.crop_size

    test_data_transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        normalize])  # what about horizontal flip

    test_full_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize])  # what about horizontal flip

    test_set = DataSet(data_dir, test_list, test_data_transform, test_full_transform)
    test_loader = DataLoader(dataset=test_set, num_workers=args.workers, collate_fn=vg_collate,
                             batch_size=args.batch_size, shuffle=False)
    return test_loader


def generate_graph(rel_num):
    numNode = rel_num
    edge_index = []
    if numNode != 1:
        for i in range(numNode):
            for j in range(numNode):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        edge_index = edge_index.t().contiguous()
    else:
        edge_index = [[], []]
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        edge_index = edge_index.contiguous()
    return edge_index


def validate(batch_data, model):
    model.eval()  
    tp, p, r = {}, {}, {}  

    name, union, obj1, obj2, bpos, target, full_im, img_rel_num, language = batch_data
    target = target.cuda()
    union_var = torch.autograd.Variable(union).cuda()
    obj1_var = torch.autograd.Variable(obj1).cuda()
    obj2_var = torch.autograd.Variable(obj2).cuda()
    bpos_var = torch.autograd.Variable(bpos).cuda()
    full_im_var = torch.autograd.Variable(full_im).cuda()

    edge_index = generate_graph(img_rel_num[0])
    count = copy.deepcopy(img_rel_num[0])
    for rel_num in img_rel_num[1:]:
        edge_index = torch.cat((edge_index, generate_graph(rel_num) + count), dim=1)
        count += rel_num

    img_rel_num = torch.autograd.Variable(img_rel_num).cuda()
    edge_index = torch.autograd.Variable(edge_index).cuda()

    target_var = torch.autograd.Variable(target)
    with torch.no_grad():
        output = model(union_var, obj1_var, obj2_var, bpos_var, full_im_var, img_rel_num, edge_index, language, target_var)

    output_f = F.softmax(output, dim=1)
    output_np = output_f.data.cpu().numpy()
    pre = np.argmax(output_np, 1)
    t = target_var.data.cpu().numpy()

    for i, item in enumerate(t):
        if item in r:
            r[item] += 1
        else:
            r[item] = 1
        if pre[i] in p:
            p[pre[i]] += 1
        else:
            p[pre[i]] = 1
        if pre[i] == item:
            if item in tp:
                tp[item] += 1
            else:
                tp[item] = 1

    precision = {}
    recall = {}
    tp_total = 0

    for k in tp.keys():
        precision[k] = float(tp[k]) / float(p[k])
        recall[k] = float(tp[k]) / float(r[k])
        tp_total += tp[k]

    p_total = 0

    for k in p.keys():
        p_total += p[k]
    precision_total = float(tp_total) / float(p_total)

    return precision_total, recall


def init_network(net, num_class):
    # Initialize the network.
    if net == 'GGNN_attn':
        from network import network

    model = network(num_class)
    return model


if __name__ == '__main__':
    val_loader = get_test_set(data_dir=data_dir, test_list=test_list)
    model = init_network(args.network, num_class)
    trained_model = torch.load(args.model_file)
    model_dict = model.state_dict()
    trained_model = {k.replace('module.', ''): v for k, v in
                     trained_model['state_dict'].items()}
    trained_model_dict = {k: v for k, v in trained_model.items() if k in model_dict}
    model_dict.update(trained_model_dict)
    model.load_state_dict(model_dict)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    criterion = nn.CrossEntropyLoss().cuda()

    acc, recall = validate(val_loader, model)
    print(acc, recall)