import os
import sys
import math
import pickle
import datetime
import cv2 as cv
import numpy as np
from enum import Enum
import HyperParameters
from sklearn.svm import LinearSVC
from scipy.ndimage import convolve
from sklearn.metrics import roc_curve, auc


def concat_images(pred, ground_truth):
    """
    :param input_image: imaginea grayscale (canalul L din reprezentarea Lab).
    :param pred: imaginea prezisa.
    :param ground_truth: imaginea ground-truth.
    :return: concatenarea imaginilor.
    """
    h, w, _ = pred.shape
    space_btw_images = int(0.2 * w)
    image = np.ones((h, w * 2 + 2 * space_btw_images, 3)) * 255
    # add ground truth
    image[:, :w] = ground_truth
    # add predicted
    offset = w + space_btw_images
    image[:, offset: offset + w] = pred
    return np.uint8(image)


def create_flow(image):
    mag = image[:, :, 0]
    angle = image[:, :, 1]
    max_flow = 64
    n = 8
    im_h = np.mod(angle / (2 * np.pi) + 1.0, 1.0)
    im_s = np.clip(mag * n / max_flow, 0, 1)
    im_v = np.clip(n - im_s, 0, 1)

    im_hsv = np.stack([im_h, im_s, im_v], 2)
    outimageHSV = np.uint8(im_hsv * 255)
    outimageBGR = cv.cvtColor(outimageHSV, cv.COLOR_HSV2BGR)
    return outimageBGR


class TemporalFrame:

    def __init__(self, temporal_size, max_size):# 15 31
        self.temporal_size = temporal_size
        self.max_size = max_size
        self.frames = []

    def add(self, frame):
        self.frames.append(frame.copy())
        if len(self.frames) > self.max_size:
            self.frames.pop(0)

    def get(self, index):
        if index < 0:
            return self.frames[self.temporal_size + index].copy()
        if index >= 0:
            return self.frames[self.temporal_size + index].copy()

    def get_middle_frame(self):
        return self.frames[self.temporal_size].copy()


def crop_bbox(img, bbox):#TODO：xmin，ymin，xmax，ymax
    crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()  #TODO：y，x
    return crop


def crop_context(current_frame, bbox, context_len):
    h, w = current_frame.shape[:2]
    h_object = bbox[3] - bbox[1]
    w_object = bbox[2] - bbox[0]

    new_xmin = bbox[0] - context_len

    padding_xmin = 0
    if new_xmin < 0:
        padding_xmin = -new_xmin
        new_xmin = 0

    new_xmax = bbox[2] + context_len

    padding_xmax = w_object + 2 * context_len
    if new_xmax > w:
        padding_xmax = w - new_xmax
        new_xmax = w

    new_ymin = bbox[1] - context_len

    padding_ymin = 0
    if new_ymin < 0:
        padding_ymin = -new_ymin
        new_ymin = 0

    new_ymax = bbox[3] + context_len

    padding_ymax = h_object + 2 * context_len

    if new_ymax > h:
        padding_ymax = h - new_ymax
        new_ymax = h

    crop = crop_bbox(current_frame, [new_xmin, new_ymin, new_xmax, new_ymax])

    padded_picture = np.zeros((h_object + 2 * context_len, w_object + 2 * context_len, 3), np.uint8)

    padded_picture[padding_ymin:padding_ymax, padding_xmin:padding_xmax] = crop

    return padded_picture


def create_dir(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)


def check_file_existence(file_path):
    return os.path.exists(file_path)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x)/sum(np.exp(x))


def train_linear_svm(x_train, labels, c):
    model = LinearSVC(penalty='l2', loss='squared_hinge', C=c, random_state=12)
    model.fit(x_train, labels)

    return model


def get_extension(file_name):
    if type(file_name) is str:
        return file_name.split('.')[-1]
    return None


def get_file_name(file_name):
    if type(file_name) is str:
        file_short_name, file_extension = os.path.splitext(file_name)
        return file_short_name
    return None

def normalize_(err_):
    err_ = np.array(err_)
    err_ = err_ - min(err_)
    err_ = err_ / max(err_)
    return err_


def predict_anomaly_on_frames(video_info_path, filter_3d, filter_2d):
    video_normality_scores = np.loadtxt(os.path.join(video_info_path, "anormality_scores.txt"))
    video_loc_v3 = np.load(os.path.join(video_info_path, "loc_v3.npy"))  # TODO：多个5个的，其中的5个，有帧，有bbox
    video_meta_data = pickle.load(open(os.path.join(video_info_path, "video_meta_data.pkl"), 'rb'))
    video_height = video_meta_data["height"]
    video_width = video_meta_data["width"]

    block_scale = HyperParameters.block_scale
    block_h = int(round(video_height / block_scale))
    block_w = int(round(video_width / block_scale))

    anomaly_scores = video_normality_scores - min(video_normality_scores)
    anomaly_scores = anomaly_scores / max(anomaly_scores)

    num_frames = video_meta_data["num_frames"]  # TODO：180
    num_bboxes = len(anomaly_scores)  # TODO：3464

    ab_event = np.zeros((block_h, block_w, num_frames))
    for i in range(num_bboxes):
        loc_V3 = np.int32(video_loc_v3[i])

        ab_event[int(round(loc_V3[2] / block_scale)): int(round(loc_V3[4] / block_scale)) + 1,
        int(round(loc_V3[1] / block_scale)): int(round(loc_V3[3] / block_scale) + 1),
        loc_V3[0]] = np.maximum(
            ab_event[int(round(loc_V3[2] / block_scale)):int(round(loc_V3[4] / block_scale)) + 1,
            int(round(loc_V3[1] / block_scale)): int(round(loc_V3[3] / block_scale)) + 1,
            loc_V3[0]], anomaly_scores[i])

    dim = 9
    filter_3d = np.ones((dim, dim, dim)) / (dim ** 3)
    ab_event3 = convolve(ab_event, filter_3d)  # ab_event.copy() #
    np.save(os.path.join(video_info_path, 'ab_event3.npy'), ab_event3)
    frame_scores = np.zeros(num_frames)
    for i in range(num_frames):
        frame_scores[i] = ab_event3[:, :, i].max()

    padding_size = len(filter_2d) // 2
    # np.savetxt('anomaly_on_frames/' + video_info_path.split(os.sep)[-1] + '.txt', frame_scores)
    # in_ = np.concatenate((np.zeros(padding_size), frame_scores, np.zeros(padding_size)))
    in_ = np.concatenate((frame_scores[:padding_size], frame_scores, frame_scores[-padding_size:]))
    frame_scores = np.correlate(in_, filter_2d, 'valid')
    return frame_scores


def gaussian_filter_3d(sigma=1.0):
    x = np.array([-2, -1, 0, 1, 2])
    f = np.exp(- (x ** 2) / (2 * (sigma ** 2))) / (sigma * np.sqrt(2 * np.pi))
    f += (1 - np.sum(f)) / len(f)
    k = np.expand_dims(f, axis=1).T * np.expand_dims(f, axis=1)
    k3d = np.expand_dims(k, axis=2).T * np.expand_dims(np.expand_dims(f, axis=1), axis=2)
    # k3d = k3d * 3
    return k3d


def gaussian_filter_(support, sigma):
    mu = support[len(support) // 2 - 1]
    filter = 1.0 / (sigma * np.sqrt(2 * math.pi)) * np.exp(-0.5 * ((support - mu) / sigma) ** 2)
    return filter


def calculate_AUC():
    filter_3d = gaussian_filter_3d(sigma=25)  # don't use it here
    filter_2d = gaussian_filter_(np.arange(1, 50), 20)

    # list all the testing videos
    videos_features_base_dir = os.path.join(HyperParameters.output_folder_base, HyperParameters.database_name,
                                            'testing')  # TODO：'ped2_output_yolo_0.50\\ped2\\testing'
    testing_videos_names = [name for name in os.listdir(videos_features_base_dir) if
                            os.path.isdir(os.path.join(videos_features_base_dir, name))]
    testing_videos_names.sort()  # TODO：['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    all_frame_scores = []
    all_gt_frame_scores = []
    roc_auc_videos = []

    for video_name in testing_videos_names:
        print(video_name)
        video_scores = predict_anomaly_on_frames(os.path.join(videos_features_base_dir, video_name), filter_3d,
                                                        filter_2d)
        all_frame_scores = np.append(all_frame_scores, video_scores)
        # read the ground truth scores at frame level
        gt_scores = np.loadtxt(os.path.join(videos_features_base_dir, video_name, "ground_truth_frame_level.txt"))
        all_gt_frame_scores = np.append(all_gt_frame_scores, gt_scores)
        fpr, tpr, _ = roc_curve(np.concatenate(([0], gt_scores, [1])), np.concatenate(([0], video_scores, [1])))
        roc_auc = auc(fpr, tpr)
        print(roc_auc)
        roc_auc_videos.append(roc_auc)
    # plt.plot(all_gt_frame_scores)
    # plt.plot(all_frame_scores)
    # plt.show()

    fpr, tpr, _ = roc_curve(all_gt_frame_scores, all_frame_scores)
    roc_auc = auc(fpr, tpr)
    print("Frame-based AUC is %.3f on %s (all data set)." % (roc_auc, HyperParameters.database_name))
    print("Avg. (on video) frame-based AUC is %.3f on %s." % (
    np.array(roc_auc_videos).mean(), HyperParameters.database_name))


