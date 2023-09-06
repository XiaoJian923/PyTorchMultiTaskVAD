## Anomaly Detection in Video via Self-Supervised and Multi-Task Learning
### Mariana-Iuliana Georgescu, Antonio Barbalau, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu, Mubarak Shah
### IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.
# 说明：该文的PyTorch实现
	原文代码是TensorFlow版本，此PyTorch版本是依据原作者分享的源码所改；附environment.yaml文件为torch版本的选择提供参考；此README文件的其余部分，为原作者所作，笔者（XJ）未对其进行删改

### Required libraries
- tested with Python 3.6 and 3.7
- numpy (tested with version 1.18.5 and 1.19.1)
- tensorflow (tested with tf1.13, tf1.14 and tf1.15)
- opencv (tested with 4.5.1)
- tested on Linux OS and Windows OS 

### Required pre-trained models
- pre-trained YoloV3 taken from [here](https://github.com/wizyoung/YOLOv3_TensorFlow)

### Set the input and output paths in the ```args.py``` file.
    '''
    database_name = 'ped2'
    output_folder_base = '/media/lili/SSD2/datasets/abnormal_event/ped2/output_yolo_%.2f' % detection_threshold
    input_folder_base = '/media/lili/SSD2/datasets/abnormal_event/ped2'
    '''

### Set the temporal size in the ```args.py``` file.
    - to temporal_size = 15 for training (we need more context for the intermittent sequences).   
    - to temporal_size = 3 for testing. 

### Run ```train.py``` 
It requires the yolov3 model in the folder models/yolov3/yolov3.ckpt

Do not forget to change the ```temporal_size``` to 15.

### Run ```test.py```
Do not forget to change the ```temporal_size``` to 3.

#### Evaluation
- The evaluation code requires to have the ground-truth frame level annotation in args.output_folder_base/test/video_name/ground_truth_frame_level.txt
- In order to write the ground-truth frame-level for UCSD Ped2, run ```other_code/write_gt_frame_level.py```, change the video_dir path accordingly.
- The 1D and 3D filters (compute_performance_scores.py lines 161 and 142) are fine-tuned for the UCSD Ped2 dataset.
- For the ShanghaiTech data set, we do not use the 3D filter.
- For the Avenue data set we do not normalize the scores for each task (compute_performance_scores.py lines 75-78).
- We used the following hyper-parameters:
  
	Avenue - block-size=20, 3D mean filter with window size = 9, 1D gaussian filter=gaussian_filter_(np.arange(1, 302), 25)
  
	Ped2 - block-size=1 (3), 3D mean filter with window size = 9, 1D gaussian filter=gaussian_filter_(np.arange(1, 50), 20)
  
	ShanghaiTech - block-size=20, 1D gaussian filter=gaussian_filter_(np.arange(1, 202), 31)
