import os
import scipy.io
import numpy as np
import HyperParameters


mat = scipy.io.loadmat('ped2.mat')
gt = mat['gt']

output_folder = os.path.join('/home/y212202033/lulu/' + HyperParameters.output_folder_base, HyperParameters.database_name, 'testing', "%s",
                             "ground_truth_frame_level.txt")

video_names = ['%02d' % i for i in range(1, 13)]


for idx, file_name in enumerate(video_names):

    video_dir = 'dataset/ped2/testing/frames/' + file_name  # os.path.join(args.input_folder_base, ProcessingType.TEST.value, "frames", file_name)
    print(video_dir)
    video_names = [f for f in os.listdir(video_dir)]
    print(len(video_names))
    new_content = np.zeros((len(video_names)))
    start_anomaly = gt[0][idx][0][0] - 1
    end_anomaly = gt[0][idx][1][0] 
    new_content[start_anomaly: end_anomaly] = 1
    np.savetxt(output_folder % file_name, new_content) 
    print(output_folder % file_name)
