import os
import glob
import cv2
import utilss
import numpy as np
import HyperParameters
from sklearn.utils import shuffle
from torch.utils.data import Dataset

class DataSetReader(Dataset):
    def __init__(self, sample_names):
        self.sample_names = shuffle(sample_names)
        self.len_seq = 7
        self.num_samples = len(self.sample_names)
        self.end_index = 0
        self.middle_frame_idx = 3
        self.negative_permutation = np.arange(self.len_seq)[::-1]

    def __len__(self):
        return len(self.sample_names)

    def generate_positions(self):
        positions = []
        num_pos = 6
        for _ in range(num_pos):
            pos = np.random.randint(1, 4)
            positions.append(pos)

        num_left = int(num_pos / 2)
        left_part = np.array(positions[:num_left]) * -1

        for i in range(num_left - 1):
            left_part[i] = left_part[i] + np.sum(left_part[i + 1:])

        right_part = positions[num_left:]
        for i in range(1, num_left):
            right_part[i] += right_part[i - 1]

        new_pos = np.array(list(left_part) + [0] + right_part) + 15

        return new_pos

    def read_samples(self, file_path):
        '''#TODO：3种数据，4种标签
        # samples = []#TODO：没有了中间帧的6帧
        # samples_consecutive = []#TODO：跳序的7帧
        # labels_detector = []#TODO：只有中间帧
        # labels_fwd_bwd = []#TODO：正序与逆序
        # labels_consecutive = []#TODO：跳序
        # samples_resnet = []#TODO：7帧
        # labels_resnet = []#TODO：加载1080维度，就是resnet50的最后一层1000维度 + YOLO的80类别one-hot
        '''

        #TODO：都是裁剪成_64的图片cube
        full_sample_cons = np.load(file_path)
        full_sample = full_sample_cons[12:19]#TODO：7帧

        samples_resnet = full_sample
        # load label  TODO：加载1000维度，就是resnet50的最后一层
        label_resnet = np.load(file_path.replace(HyperParameters.samples_folder_name, HyperParameters.imagenet_logits_folder_name).replace('_64.npy', '.npy'))

        meta = np.loadtxt(file_path.replace(HyperParameters.samples_folder_name, HyperParameters.meta_folder_name).replace('_64.npy', '.txt'))
        yolo_logits = np.zeros(80)
        yolo_logits[int(meta[-2]) - 1] = meta[-1]#TODO：这个得到的是yolo目标检测的判别标签，可是不是yolo真实的80个类别的概率，而是one-hot
        logits_resnet = np.maximum(np.squeeze(label_resnet), 0)#TODO：np.squeeze删除某一维度  np.maximum留下最大的元素

        labels_resnet = np.concatenate((logits_resnet, yolo_logits))

        if np.random.rand() >= 0.5:#TODO：这个是前向
            labels_fwd_bwd = [1, 0]
        else:
            labels_fwd_bwd = [0, 1]
            full_sample = full_sample[self.negative_permutation]#TODO：这个是逆序了

        if np.random.rand() >= 0.5:#TODO：这个是正常连续运动顺序，中间帧是当前帧
            labels_consecutive = [1, 0]
            samples_consecutive = full_sample
        else:
            positions = self.generate_positions()
            sample_ = full_sample_cons[positions]#TODO：中间帧不变，前后都是跳帧序，也是7帧
            samples_consecutive = sample_
            labels_consecutive = [0, 1]

        label = full_sample[self.middle_frame_idx]#TODO：取出中间的图片帧bbox
        sample = np.delete(full_sample, self.middle_frame_idx, 0)#TODO：按行删除，删去中间帧

        samples = sample#TODO：没有了中间帧的6帧
        labels_detector = label#TODO：只有中间帧
        samples = samples.transpose(3,0,1,2).astype(np.float32)
        samples_consecutive = samples_consecutive.transpose(3,0,1,2).astype(np.float32)
        samples_resnet = samples_resnet.transpose(3,0,1,2).astype(np.float32)
        labels_detector = labels_detector.transpose(2,0,1).astype(np.float32)

        return samples, samples_consecutive, samples_resnet,  labels_detector, labels_fwd_bwd, labels_consecutive, labels_resnet

    def __getitem__(self, index):
        if self.end_index == self.num_samples:
            self.end_index = 0
            self.sample_names = shuffle(self.sample_names)

        self.end_index += 1
        self.end_index = min(self.end_index, self.num_samples)

        names = self.sample_names[index]
        # TODO：6帧图片——缺少中间帧，也用于前序或逆序；7帧连续或跳序图片；7帧完全的图片
        # TODO：取3种数据，4种标签
        # TODO：1张帧；1个正逆序标签；1个连续跳序标签；1个1080维的模型蒸馏标签
        samples, samples_consecutive, samples_resnet, labels_detector, labels_fwd_bwd, labels_consecutive, labels_resnet = self.read_samples(names)
        samples = np.array(samples) / 255.0
        samples_consecutive = np.array(samples_consecutive) / 255.0
        samples_resnet = np.array(samples_resnet) / 255.0

        labels_detector = np.array(labels_detector) / 255.0
        labels_fwd_bwd = np.array(labels_fwd_bwd)
        labels_consecutive = np.array(labels_consecutive)
        return samples, samples_consecutive, samples_resnet, labels_detector, labels_fwd_bwd, labels_consecutive, labels_resnet


def create_readers_train_val_split():
    folder_base = os.path.join(HyperParameters.output_folder_base, HyperParameters.database_name, utilss.ProcessingType.TRAIN.value)
    videos_names = os.listdir(folder_base)
    names_train = []
    names_val = []
    for video_name in videos_names:
        video_samples = glob.glob(os.path.join(folder_base, video_name, HyperParameters.samples_folder_name, '*_64.npy'))
        video_samples.sort()
        num_examples = len(video_samples)
        num_training = int(0.85 * num_examples)
        names_train += video_samples[:num_training]
        names_val += video_samples[num_training:]

    print('num training examples', len(names_train))
    print('num validation examples', len(names_val))

    reader_train = DataSetReader(names_train)
    reader_val = DataSetReader(names_val)

    return reader_train, reader_val


class DataSetTester(Dataset):
    def __init__(self, video_name):
        self.video_name = video_name
        self.meta_base_dir = os.path.join(HyperParameters.output_folder_base, HyperParameters.database_name, 'testing', "%s", HyperParameters.meta_folder_name, "%s")
        self.video_patch_base_dir = os.path.join(HyperParameters.output_folder_base, HyperParameters.database_name, 'testing', "%s", HyperParameters.samples_folder_name, '%s')

        self.loc_v3_path = os.path.join(HyperParameters.output_folder_base, HyperParameters.database_name, 'testing', "%s", "loc_v3.npy")

        self.sample_names = os.listdir(self.meta_base_dir % (self.video_name, ""))  # TODO:这里都是.txt  # maybe this is a hack :)



    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, index):
        sample_name = self.sample_names[index]
        video_patch = np.load(self.video_patch_base_dir % (self.video_name, sample_name.replace('.txt', '.npy')))#7,31,16,3
        meta = np.loadtxt(self.meta_base_dir % (self.video_name, sample_name))# 帧，bbox，类别，置信分数

        resized_sample = []
        for i in range(len(video_patch)):
            sample_obj = video_patch[i]  # TODO：31,16,3
            sample_obj = cv2.resize(sample_obj, (64, 64), interpolation=cv2.INTER_CUBIC)  # TODO：64,64,3
            resized_sample.append(sample_obj)
        full_sample = np.array(resized_sample).astype(np.float32) / 255.0  # TODO：7,64,64,3
        label = full_sample[HyperParameters.temporal_size]# TODO：64,64,3
        sample = np.delete(full_sample, HyperParameters.temporal_size, 0)  # TODO：6,64,64,3 删除中间的帧
        full_sample = full_sample.transpose(3, 0, 1, 2)#3,7,64,64
        label = label.transpose(2, 0, 1)#3,64,64
        sample = sample.transpose(3, 0, 1, 2)#3,6,64,64  TODO：这里目前都是ndarray类型，但经过DataLoader之后，就会变成Tensor类型
        return sample, full_sample, label, meta


def create_readers_test(video_name):
    reader = DataSetTester(video_name)
    return reader