import json
import h5py
import os
import pickle
from PIL import Image
import io
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import random
import copy
import csv

class AV_KS_Dataset(Dataset):

    def __init__(self, mode, transforms=None):
        self.data = []
        self.label = []
        # 训练、验证、测试集的数据路径设定
        # train, val, test 的 visual_path 都是相同的，但是 csv_path 不同，对应读取的数据应该也不同
        if mode=='train':
            csv_path = '..\\data\\tiny-Kinetics-400\\tiny-kinetics-400\\annotations\\tiny_train.csv'
            self.audio_path = '..\\data\\tiny-Kinetics-400\\tiny-Kinetics-400-audio\\train_spec'
            self.visual_path = '..\\data\\tiny-Kinetics-400\\tiny-Kinetics-400-30fps-frames'

        elif mode=='val':
            csv_path = '..\\data\\tiny-Kinetics-400\\tiny-kinetics-400\\annotations\\tiny_val.csv'
            self.audio_path = '..\\data\\tiny-Kinetics-400\\tiny-Kinetics-400-audio\\test_spec'
            self.visual_path = '..\\data\\tiny-Kinetics-400\\tiny-Kinetics-400-30fps-frames'

        else:
            csv_path = '..\\data\\tiny-Kinetics-400\\tiny-kinetics-400\\annotations\\tiny_val.csv'
            self.audio_path = '..\\data\\tiny-Kinetics-400\\tiny-Kinetics-400-audio\\test_spec'
            self.visual_path = '..\\data\\tiny-Kinetics-400\\tiny-Kinetics-400-30fps-frames'

        # TODO: 修改读取的方式，我们的 tiny_train.csv 毕竟和原始代码期望的很不一样
        # 因为还要保存标签，所以只能用 annotation 里的索引文件
        # 这里只获取音频频谱的数据
        '''
        数据长这样：
        label,youtube_id,time_start,time_end,split,is_cc
        abseiling,_4YTwq0-73Y,44,54,train,0
        "air drumming",_axE99QAhe8,26,36,train,0
        '''
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader)
            rows = list(reader)
            for i, row in enumerate(rows):
                label_id = row[-1].replace(' ', '_')
                name = row[1] + '_' + row[2].zfill(6) + '_' + row[3].zfill(6)
                if os.path.exists(os.path.join(self.audio_path, name + '.wav' + '.npy')):
                    self.data.append(name)
                    self.label.append(label_id)

        # with open(csv_path) as f:
        #     for line in f:
        #         item = line.split("\n")[0].split(" ")
        #         name = item[0].split("/")[-1]
        #         # 获取数据集
        #         if os.path.exists(self.audio_path + '/' + name + '.npy'):
        #             self.data.append(name)
        #             self.label.append(int(item[-1]))


        print('data load finish')

        self.mode = mode
        self.transforms = transforms

        self._init_atransform()

        print('# of files = %d ' % len(self.data))
    # tranform 添加数据增强
    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        av_file = self.data[idx]

        spectrogram = np.load(self.audio_path + '/' + av_file + '.npy')
        spectrogram = np.expand_dims(spectrogram, axis=0)
        
        # Visual
        path = self.visual_path + '/' + av_file
        file_num = len([lists for lists in os.listdir(path)])

        if self.mode == 'train':
            # 图像变换
            transf = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transf = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        # 对每个样本，从视频中选取 3 个图像帧（随机采样）
        pick_num = 3
        seg = int(file_num / pick_num)
        path1 = []
        image = []
        image_arr = []
        t = [0] * pick_num

        for i in range(pick_num):
            if self.mode == 'train':
                # 训练时随机采样
                t[i] = random.randint(i * seg + 1, i * seg + seg) if file_num > 6 else 1
                if t[i] >= 10:
                    t[i] = 9
            else:
                # 测试时取中间帧
                t[i] = i*seg + max(int(seg/2), 1) if file_num > 6 else 1

            path1.append('frame_0000' + str(t[i]) + '.jpg')
            image.append(Image.open(path + "/" + path1[i]).convert('RGB'))
            # 将多个图像帧变换后进行融合
            image_arr.append(transf(image[i]))
            image_arr[i] = image_arr[i].unsqueeze(1).float()
            if i == 0:
                image_n = copy.copy(image_arr[i])
            else:
                image_n = torch.cat((image_n, image_arr[i]), 1)
        

        label = self.label[idx]
        # 返回融合后的图像，频谱，标签，索引
        return  image_n,spectrogram,label,idx


class AV_KS_Dataset_sample_level(Dataset):

    def __init__(self, mode, contribution, transforms=None):
        self.data = []
        self.label = []
        self.drop = []
        
        if mode=='train':
            csv_path = 'ks_audio/train_1fps_path.txt'
            self.audio_path = 'ks_audio/train/'
            self.visual_path = 'ks_visual/train/'
        
        elif mode=='val':
            csv_path = 'ks_audio/val_1fps_path.txt'
            self.audio_path = 'ks_audio/val/'
            self.visual_path = 'ks_visual/val/'

        else:
            csv_path = 'ks_audio/test_1fps_path.txt'
            self.audio_path = 'ks_audio/test/'
            self.visual_path = 'ks_visual/test/'


        with open(csv_path) as f:
            for line in f:
                item = line.split("\n")[0].split(" ")
                name = item[0].split("/")[-1]

                if os.path.exists(self.audio_path + '/' + name + '.npy'):
                    self.data.append(name)
                    self.label.append(int(item[-1]))
                    self.drop.append(0)

        print('data load finish')
        length = len(self.data)

        # visual = 2, audio = 1, none = 0
        # drop 表示训练时要丢弃对应模态。
        # 丢弃是为了计算单个模态的预测结果，最终为计算每个模态的最终贡献服务
        # 不同之处在于，使用 contribution 进行重采样
        for i in range(length):
          contrib_a, contrib_v = contribution[i]
          # 论文中的单调递增函数，代码实现居然只是分段。不过这样确实好写
          if 0.4 < contrib_a < 1: #0.5
                for tt in range(1):
                        self.data.append(self.data[i])
                        self.label.append(self.label[i])
                        self.drop.append(2)
          elif -0.1 < contrib_a < 0.4: #0.0
              for tt in range(2):
                      self.data.append(self.data[i])
                      self.label.append(self.label[i])
                      self.drop.append(2)
          elif contrib_a <-0.1: #-0.5
              for tt in range(3):
                      self.data.append(self.data[i])
                      self.label.append(self.label[i])
                      self.drop.append(2)

          if 0.4 < contrib_v < 1: #0.5
              for tt in range(1):
                      self.data.append(self.data[i])
                      self.label.append(self.label[i])
                      self.drop.append(1)
          elif -0.1 < contrib_v < 0.4:
              for tt in range(2):
                      self.data.append(self.data[i])
                      self.label.append(self.label[i])
                      self.drop.append(1)
          elif contrib_v < -0.1:
              for tt in range(3):
                      self.data.append(self.data[i])
                      self.label.append(self.label[i])
                      self.drop.append(1)

        print('data resample finish')

        self.mode = mode
        self.transforms = transforms

        self._init_atransform()

        print('# of files = %d ' % len(self.data))

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        av_file = self.data[idx]

        spectrogram = np.load(self.audio_path + '/' + av_file + '.npy')
        spectrogram = np.expand_dims(spectrogram, axis=0)
        
        # Visual
        path = self.visual_path + '/' + av_file
        file_num = len([lists for lists in os.listdir(path)])

        if self.mode == 'train':

            transf = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transf = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        pick_num = 3
        seg = int(file_num / pick_num)
        path1 = []
        image = []
        image_arr = []
        t = [0] * pick_num

        for i in range(pick_num):
            if self.mode == 'train':
                t[i] = random.randint(i * seg + 1, i * seg + seg) if file_num > 6 else 1
                if t[i] >= 10:
                    t[i] = 9
            else:
                t[i] = i*seg + max(int(seg/2), 1) if file_num > 6 else 1

            path1.append('frame_0000' + str(t[i]) + '.jpg')
            image.append(Image.open(path + "/" + path1[i]).convert('RGB'))

            image_arr.append(transf(image[i]))
            image_arr[i] = image_arr[i].unsqueeze(1).float()
            if i == 0:
                image_n = copy.copy(image_arr[i])
            else:
                image_n = torch.cat((image_n, image_arr[i]), 1)
        

        label = self.label[idx]
        drop = self.drop[idx]

        return  image_n,spectrogram,label,idx,drop

class AV_KS_Dataset_modality_level(Dataset):

    def __init__(self, mode, contribution_a, contribution_v, alpha, func='linear', transforms=None):
        self.data = []
        self.label = []
        self.drop = []
        
        if mode=='train':
            csv_path = 'ks_audio/train_1fps_path.txt'
            self.audio_path = 'ks_audio/train/'
            self.visual_path = 'ks_visual/train/'
        
        elif mode=='val':
            csv_path = 'ks_audio/val_1fps_path.txt'
            self.audio_path = 'ks_audio/val/'
            self.visual_path = 'ks_visual/val/'

        else:
            csv_path = 'ks_audio/test_1fps_path.txt'
            self.audio_path = 'ks_audio/test/'
            self.visual_path = 'ks_visual/test/'


        with open(csv_path) as f:
            for line in f:
                item = line.split("\n")[0].split(" ")
                name = item[0].split("/")[-1]

                if os.path.exists(self.audio_path + '/' + name + '.npy'):
                    self.data.append(name)
                    self.label.append(int(item[-1]))
                    self.drop.append(0)

        print('data load finish')
        length = len(self.data)

        #drop visual = 2, audio = 1, none = 0
        
        gap_a = 1.0 - contribution_a
        gap_v = 1.0 - contribution_v

        if func == 'linear':
            difference = (abs(gap_a - gap_v) / 3 * 2 ) * alpha
        elif func == 'tanh':
            tanh = torch.nn.Tanh()
            difference = tanh(torch.tensor((abs(gap_a - gap_v) / 3 * 2 ) * alpha))
        elif func == 'square':
            difference = (abs(gap_a - gap_v) / 3 * 2 ) ** 1.5 * alpha
        resample_num = int(difference * length)
        sample_choice = np.random.choice(length, resample_num)

        for i in sample_choice:
            self.data.append(self.data[i])
            self.label.append(self.label[i])
            if gap_a > gap_v:
                self.drop.append(2)
            else:
                self.drop.append(1)

        print('data resample finish')

        self.mode = mode
        self.transforms = transforms

        self._init_atransform()

        print('# of files = %d ' % len(self.data))

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        av_file = self.data[idx]

        spectrogram = np.load(self.audio_path + '/' + av_file + '.npy')
        spectrogram = np.expand_dims(spectrogram, axis=0)
        
        # Visual
        path = self.visual_path + '/' + av_file
        file_num = len([lists for lists in os.listdir(path)])

        if self.mode == 'train':

            transf = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transf = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        pick_num = 3
        seg = int(file_num / pick_num)
        path1 = []
        image = []
        image_arr = []
        t = [0] * pick_num

        for i in range(pick_num):
            if self.mode == 'train':
                t[i] = random.randint(i * seg + 1, i * seg + seg) if file_num > 6 else 1
                if t[i] >= 10:
                    t[i] = 9
            else:
                t[i] = i*seg + max(int(seg/2), 1) if file_num > 6 else 1

            path1.append('frame_0000' + str(t[i]) + '.jpg')
            image.append(Image.open(path + "/" + path1[i]).convert('RGB'))

            image_arr.append(transf(image[i]))
            image_arr[i] = image_arr[i].unsqueeze(1).float()
            if i == 0:
                image_n = copy.copy(image_arr[i])
            else:
                image_n = torch.cat((image_n, image_arr[i]), 1)
        

        label = self.label[idx]
        drop = self.drop[idx]

        return  image_n,spectrogram,label,idx,drop
