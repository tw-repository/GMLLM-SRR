import os, sys
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import pickle as pkl


def adjust_box_coordinates(box):
    left, upper, right, lower = box
    # Ensure left < right
    if left > right:
        left, right = right, left
    # Ensure upper < lower
    if upper > lower:
        upper, lower = lower, upper
    return left, upper, right, lower


class DataSet(data.Dataset):
    def __init__(self, image_dir, data_dir, input_transform, full_im_transform):

        self.input_transform = input_transform
        self.full_im_transform = full_im_transform

        self.image_dir = image_dir
        self.person_box = []
        self.person_pair = []
        self.object_box = []
        self.rel_classes = []
        self.rel_coarse_classes = []
        self.object_classes = []
        self.img_names = []
        self.language = []

        data = pkl.load(open(data_dir, "rb"), encoding='latin1')

        for item in data:
            self.img_names.append(item['img'])
            self.person_box.append(item['Pbboxs'])
            self.person_pair.append(item['person_pair'])
            self.rel_classes.append(item['rel'])
            self.language.append(item['language'])

    def __getitem__(self, index):

        # PISC
        # bounding box
        bbox_min = 0
        # bbox_max = 1497
        bbox_m = 1497.

        area_min = 198
        # area_max = 939736
        area_m = 939538.

        img = Image.open(os.path.join(self.image_dir, str(self.img_names[index]))).convert('RGB')  # convert gray to rgb

        person_box = self.person_box[index]
        ## target
        rel_classes = self.rel_classes[index]
        name = self.img_names[index]
        person_num = len(person_box)
        person_pair = self.person_pair[index]

        person_box1 = torch.zeros(len(rel_classes), 3, 224, 224)
        person_box2 = torch.zeros(len(rel_classes), 3, 224, 224)
        union_box = torch.zeros(len(rel_classes), 3, 224, 224)
        bposes = np.zeros((len(rel_classes), 10))

        for rel in range(len(rel_classes)):
            if person_pair.shape[0] != len(rel_classes):
                print(person_pair)

            p1 = person_pair[rel, 0]
            p2 = person_pair[rel, 1]
            # print(p1)
            p1_box = person_box[p1]
            # print(p1_box)
            p1_box = adjust_box_coordinates(p1_box)
            per1 = img.crop((p1_box[0], p1_box[1], p1_box[2], p1_box[3]))
            per1 = self.input_transform(per1)
            person_box1[rel, :] = per1

            p2_box = person_box[p2]
            p2_box = adjust_box_coordinates(p2_box)
            per2 = img.crop((p2_box[0], p2_box[1], p2_box[2], p2_box[3]))
            per2 = self.input_transform(per2)
            person_box2[rel, :] = per2

            u_x1 = min(p1_box[0], p2_box[0])
            u_y1 = min(p1_box[1], p2_box[1])
            u_x2 = max(p1_box[2], p2_box[2])
            u_y2 = max(p1_box[3], p2_box[3])
            union = img.crop((u_x1, u_y1, u_x2, u_y2))
            union = self.input_transform(union)

            union_box[rel, :] = union

            area1 = (p1_box[2] - p1_box[0] + 1) * (p1_box[3] - p1_box[1] + 1)
            area2 = (p2_box[2] - p2_box[0] + 1) * (p2_box[3] - p2_box[1] + 1)
            p1_box = list(p1_box)
            p2_box = list(p2_box)
            p1_box.append(area1)
            p2_box.append(area2)
            bpos = np.array(p1_box + p2_box, dtype=np.float32)

            # normalize
            bpos[0:4] = 2 * (bpos[0:4] - bbox_min) / bbox_m - 1
            bpos[4] = 2 * (bpos[4] - area_min) / area_m - 1
            bpos[5:9] = 2 * (bpos[5:9] - bbox_min) / bbox_m - 1
            bpos[9] = 2 * (bpos[9] - area_min) / area_m - 1
            # print("rel id===>>",rel)
            # print(bpos)
            bposes[rel, :] = bpos
        bposes = torch.from_numpy(bposes).float()

        if self.full_im_transform:
            full_im = self.full_im_transform(img)
        else:
            full_im = img

        union_box = torch.from_numpy(np.array(union_box))
        rel_num = len(rel_classes)
        rel_classes = torch.LongTensor(rel_classes)
        img_rel_num = torch.LongTensor([rel_num])

        return self.img_names[index], union_box, person_box1, person_box2, \
               bposes, rel_classes, full_im, img_rel_num, self.language[index]

    def __len__(self):
        return len(self.img_names)

