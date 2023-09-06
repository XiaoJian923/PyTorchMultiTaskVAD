import torch
import numpy as np
import glob
import os
import utilss
import HyperParameters
from torchvision import models

def extract_resnet50():

    print('Now resnet50 features extraction')
    model = models.resnet50(pretrained=True).eval()#TODO：要设置成eval()模式，否则默认就是train()模型，这时batchsize拒绝等于1的情况

    folder_base = os.path.join(HyperParameters.output_folder_base, HyperParameters.database_name, HyperParameters.folder_name)
    output_folder_base = os.path.join(HyperParameters.output_folder_base, HyperParameters.database_name, HyperParameters.folder_name,
                                      '%s', HyperParameters.imagenet_logits_folder_name)

    videos_names = os.listdir(folder_base)
    for video_name in videos_names:
        print('shuchu   features_extractor    :',video_name)
        video_samples = glob.glob(os.path.join(folder_base, video_name, HyperParameters.samples_folder_name, '*.npy'))
        output_folder = output_folder_base % video_name
        utilss.create_dir(output_folder)
        for video_sample in video_samples:
            if video_sample.find('_64.npy') != -1:
                continue

            short_file_name = video_sample.split(os.sep)[-1]
            sample = np.load(video_sample)
            img = sample[15]
            x = np.expand_dims(img, axis=0).transpose(0,3,1,2).astype(np.float32)
            x = torch.from_numpy(x)
            print(x.shape)
            with torch.no_grad():
                predictions = model(x).numpy()
            np.save(os.path.join(output_folder, short_file_name), predictions)

