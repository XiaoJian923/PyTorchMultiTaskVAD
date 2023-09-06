import argparse
import numpy as np


parser = argparse.ArgumentParser(description="Xiao_Jian")
parser.add_argument('--mode', type=str, default='train', help='want to train or test')
parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs for train')
args = parser.parse_args()




#TODO#############################  数据集输入  #####################################
#TODO：所有数据集存放路径
AllDatasetPath = 'dataset'
#TODO：具体数据集名称
database_name = 'ped2'
# database_name = 'avenue'
# database_name = 'shanghaitech'
#TODO：训练集还是测试集
if args.mode == 'train':
    mode = 'train'
    temporal_size = 15
    folder_name = 'training'
else:
    mode = 'test'
    temporal_size = 3
    folder_name = 'testing'
#TODO：是否还有frames文件夹，再往下就是video_names (可选)
frames = 'frames'
# frames = ''




#TODO#############################  目标检测器  #####################################
#TODO：模型名称
# ObjectDetectionModel = 'yolov3_pytorch'
# ObjectDetectionModel = 'yolov3_tensorflow'
ObjectDetectionModel = 'yolov5'
# ObjectDetectionModel = 'faster_rcnn'
# ObjectDetectionModel = 'ssd'
#TODO：模型阈值
detection_threshold = 0.5




#TODO#############################  数据集输出  #####################################
#TODO：处理过的数据集存放路径
output_folder_base = '%s_output_%s_%.2f' % (database_name, ObjectDetectionModel, detection_threshold)
#TODO：目标检测的元数据 帧、bbox、目标类别、置信分数 存放在文本文件.txt中
meta_folder_name = 'meta_%d_%.2f' % (temporal_size, detection_threshold)
#TODO：根据目标检测得到的时空立方体，包括bbox大小的和resize后的两种，存放在.npy文件中
samples_folder_name = 'images_%d_%.2f' % (temporal_size, detection_threshold)
#TODO：存放resnet50的1000维度向量，用于模型蒸馏
imagenet_logits_folder_name = 'imagenet_logits_before_softmax'




#TODO#############################  其它超参数  #####################################
block_scale = 3
save_dir = 'checkpoint'
num_epochs = args.num_epochs
temporal_offsets = np.arange(-temporal_size, temporal_size + 1, 1)
checkpoint_path = 'checkpoint/models_epoch-3.pth.tar'