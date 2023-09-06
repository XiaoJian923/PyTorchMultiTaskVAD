import utilss
import torch
import pickle
import numpy as np
import HyperParameters
from folder_images import *

def read_frames_from_video(video, temporal_frames, num_frames):# 30
    for i in range(num_frames):
        if video.has_next:
            frame = video.read_frame()
            temporal_frames.add(frame)


def get_meta(frame_idx, detection):
    return np.array([frame_idx, int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3]), int(detection[5])+1, detection[4]])

def yolov5_object_extraction():

    print('Now yolo object extraction')

    #TODO：可以再换模型，精度可能不是很重要，重要的是bbox，，pytorch不如tensorflow
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True)
    model.conf = HyperParameters.detection_threshold


    #TODO：video路径
    video_dir = os.path.join(HyperParameters.AllDatasetPath, HyperParameters.database_name, HyperParameters.folder_name, HyperParameters.frames)

    video_names = [f for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))]
    video_names.sort()
    num_videos = len(video_names)


    for video_idx, video_name in enumerate(video_names):
        print("Processing video %s, %d/%d.." % (video_name, video_idx + 1, num_videos))

        video = FolderImage(os.path.join(video_dir, video_name))

        # create output dir
        video_patch_output_dir = os.path.join(HyperParameters.output_folder_base, HyperParameters.database_name, HyperParameters.folder_name, video.name,
                                              HyperParameters.samples_folder_name)
        utilss.create_dir(video_patch_output_dir)  # TODO：创建 ped2_output_yolo_0.50\ped2\training\01\images_15_0.50
        meta_output_dir = os.path.join(HyperParameters.output_folder_base, HyperParameters.database_name, HyperParameters.folder_name, video.name,
                                       HyperParameters.meta_folder_name)
        utilss.create_dir(meta_output_dir)  # TODO：创建 ped2_output_yolo_0.50\ped2\training\01\meta_15_0.50

        temporal_frames = utilss.TemporalFrame(temporal_size=HyperParameters.temporal_size, max_size=2 * HyperParameters.temporal_size + 1)
        read_frames_from_video(video, temporal_frames, temporal_frames.max_size - 1)  # fill the queue - 1
        frame_idx = HyperParameters.temporal_size - 1  # 14

        while video.has_next:
            # read a frame and add to queue
            frame = video.read_frame()
            if frame is None:
                break
            temporal_frames.add(frame)
            frame_idx += 1
            frame_to_process = temporal_frames.get_middle_frame()#TODO：240,360,3
            results = model(frame_to_process) #TODO：这里是得到预测结果
            detections = results.xyxy[0].numpy()
            for idx_detection, detection in enumerate(detections):
                np.savetxt(os.path.join(meta_output_dir, '%05d_%05d.txt' % (frame_idx, idx_detection)),
                           get_meta(frame_idx, detection))

                video_patch = []
                for i, temporal_offset in enumerate(HyperParameters.temporal_offsets):
                    temporal_offset_frame = temporal_frames.get(temporal_offset)
                    crop = utilss.crop_bbox(temporal_offset_frame, [int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])])
                    video_patch.append(crop)

                np.save(os.path.join(video_patch_output_dir, '%05d_%05d.npy' % (frame_idx, idx_detection)),
                        np.array(video_patch, np.uint8))

        # create video metadata
        video_meta_data = {"num_frames": video.num_frames, "height": video.height, "width": video.width}
        pickle.dump(video_meta_data, open(os.path.join(HyperParameters.output_folder_base, HyperParameters.database_name, HyperParameters.folder_name, video.name,
                                                       "video_meta_data.pkl"), "wb"))

