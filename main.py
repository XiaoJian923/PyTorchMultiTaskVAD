import os
import sys
import utilss
import numpy as np
from tqdm import tqdm
import HyperParameters
operating_system = sys.platform
from tools.object_extraction import *
from tools.features_extractor import *
from torch.utils.data import DataLoader
from backbone.torch_four_task_model import *
from tools.resize_video_patches import resize_video_patch
from tools.data_set_reader import create_readers_train_val_split, create_readers_test

# #TODO：定义模型
# model1_middle_frame = MiddleFrame()
# print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model1_middle_frame.parameters()]):,}")
# model2_forward_backward = ForwardBackward()
# print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model2_forward_backward.parameters()]):,}")
# model3_consecutive_intermittent = ConsecutiveIntermittent()
# print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model3_consecutive_intermittent.parameters()]):,}")
# model4_distill = Distill()
# print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model4_distill.parameters()]):,}")


def main_fun(mode='training'):

    #TODO：定义模型
    model1_middle_frame = MiddleFrame()
    model2_forward_backward = ForwardBackward()
    model3_consecutive_intermittent = ConsecutiveIntermittent()
    model4_distill = Distill()

    #TODO：定义数据读取
    if mode=='training':
        reader_train, reader_val = create_readers_train_val_split()
        train_dataloader = DataLoader(reader_train, batch_size=8, num_workers=0, drop_last=True)#TODO：CPU环境下，num_workers必须得是0
        val_dataloader = DataLoader(reader_val, batch_size=8, num_workers=0)#TODO：训练模式下，batchsize不允许为1
        trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}

        #TODO：定义优化器，损失函数
        optimizer = torch.optim.Adam([{'params': model1_middle_frame.parameters()},
                                      {'params': model2_forward_backward.parameters()},
                                      {'params': model3_consecutive_intermittent.parameters()},
                                      {'params': model4_distill.parameters()}])
        loss1 = torch.nn.L1Loss()
        loss2 = torch.nn.CrossEntropyLoss()
    if mode=='testing':
        #TODO：测试的数据读取
        # reader_test = create_readers_test()
        # test_dataloader = DataLoader(reader_test, batch_size=8, num_workers=0, drop_last=True)
        if operating_system.find("win") == -1:
            # linux sys
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            checkpoint = torch.load(HyperParameters.checkpoint_path)
        else:
            # windows sys
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            checkpoint = torch.load(HyperParameters.checkpoint_path, map_location=lambda storage, loc: storage)

        model1_middle_frame.load_state_dict(checkpoint['state_dict1'])
        model2_forward_backward.load_state_dict(checkpoint['state_dict2'])
        model3_consecutive_intermittent.load_state_dict(checkpoint['state_dict3'])
        model4_distill.load_state_dict(checkpoint['state_dict4'])

    if mode == 'training':
        # create save_dir
        utilss.create_dir(os.path.join(HyperParameters.save_dir, HyperParameters.database_name))
        for epoch in range(HyperParameters.num_epochs):
            print("Epoch: %d/%d" % (epoch + 1, HyperParameters.num_epochs))

            for phase in ['train', 'val']:
                if phase == 'train':
                    model1_middle_frame.train()
                    model2_forward_backward.train()
                    model3_consecutive_intermittent.train()
                    model4_distill.train()
                else:
                    model1_middle_frame.eval()
                    model2_forward_backward.eval()
                    model3_consecutive_intermittent.eval()
                    model4_distill.eval()

                for x, x_consecutive, x_resnet, y_decoder, y_fwd_bwd, y_consecutive, y_resnet in tqdm(trainval_loaders[phase]):
                    x = x
                    x_consecutive = x_consecutive
                    x_resnet = x_resnet
                    y_decoder = y_decoder
                    # y_fwd_bwd = y_fwd_bwd.cuda()
                    y_consecutive = y_consecutive
                    y_resnet = y_resnet

                    optimizer.zero_grad()

                    if phase == 'train':
                        #TODO：1
                        out_middle_frame = model1_middle_frame(x)
                        loss_middle_frame = loss1(y_decoder, out_middle_frame)

                        #TODO：2
                        out_forward_backward = model2_forward_backward(x)
                        a = np.argmax(y_fwd_bwd, 1)
                        loss_forward_backward = loss2(out_forward_backward, a)

                        #TODO: 3
                        out_consecutive = model3_consecutive_intermittent(x_consecutive)
                        b = np.argmax(y_consecutive, 1)
                        loss_consecutive = loss2(out_consecutive, b)

                        # TODO: 4
                        out_distill = model4_distill(x_resnet)
                        loss_distill = loss1(out_distill, y_resnet)
                    else:
                        with torch.no_grad():
                            # TODO：1
                            out_middle_frame = model1_middle_frame(x)
                            loss_middle_frame = loss1(y_decoder, out_middle_frame)

                            # TODO：2
                            out_forward_backward = model2_forward_backward(x)
                            a = np.argmax(y_fwd_bwd, 1)
                            loss_forward_backward = loss2(out_forward_backward, a)

                            # TODO: 3
                            out_consecutive = model3_consecutive_intermittent(x_consecutive)
                            b = np.argmax(y_consecutive, 1)
                            loss_consecutive = loss2(out_consecutive, b)

                            # TODO: 4
                            out_distill = model4_distill(x_resnet)
                            loss_distill = loss1(out_distill, y_resnet)

                    # print(phase, 'task 1: ', loss_middle_frame, 'task 2: ', loss_forward_backward, 'task 3: ', loss_consecutive, 'task 4: ', loss_distill)
                    # TODO：总损失  据此优化
                    loss_all = loss_middle_frame + loss_forward_backward + loss_consecutive + 0.5 * loss_distill
                    # print(phase, 'All_loss each batch_size: ', loss_all)

                    if phase == 'train':
                        loss_all.backward()
                        optimizer.step()

                print(phase, 'Epoch: %d/%d   All_loss : %.5f',(epoch + 1, HyperParameters.num_epochs, loss_all))

            torch.save({
                'epoch': epoch + 1,
                'state_dict1': model1_middle_frame.state_dict(),
                'state_dict2': model2_forward_backward.state_dict(),
                'state_dict3': model3_consecutive_intermittent.state_dict(),
                'state_dict4': model4_distill.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(HyperParameters.save_dir, HyperParameters.database_name, 'models' + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(os.path.join(HyperParameters.save_dir, HyperParameters.database_name, 'models' + '_epoch-' + str(epoch) + '.pth.tar')))

    if mode == 'testing':
        model1_middle_frame.eval()
        model2_forward_backward.eval()
        model3_consecutive_intermittent.eval()
        model4_distill.eval()

        videos_features_base_dir = os.path.join(HyperParameters.output_folder_base, HyperParameters.database_name, 'testing')
        # TODO:'ped2_output_yolo_0.50\\ped2\\testing'
        videos_names = os.listdir(videos_features_base_dir)
        videos_names.sort()

        concat_features_path = os.path.join(HyperParameters.output_folder_base, HyperParameters.database_name, 'testing', "%s", "anormality_scores.txt")
        # TODO:'ped2_output_yolo_0.50\\ped2\\testing\\%s\\anormality_scores.txt'
        loc_v3_path = os.path.join(HyperParameters.output_folder_base, HyperParameters.database_name, 'testing', "%s", "loc_v3.npy")
        # TODO:'ped2_output_yolo_0.50\\ped2\\testing\\%s\\loc_v3.npy'


        for video_name in videos_names:
            print('processing video : ', video_name)

            video_features_path = concat_features_path % video_name  # TODO：'ped2_output_yolo_0.50\\ped2\\testing\\01\\anormality_scores.txt'
            video_loc_v3_path = loc_v3_path % video_name  # TODO：'ped2_output_yolo_0.50\\ped2\\testing\\01\\loc_v3.npy'

            reader_test = create_readers_test(video_name)
            test_dataloader = DataLoader(reader_test, batch_size=1, num_workers=0, drop_last=True)

            video_features = []
            errs_1 = []
            errs_2 = []
            errs_3 = []
            errs_4 = []
            loc_v3_video = []

            for sample, full_sample, label, meta in tqdm(test_dataloader):  # TODO：对每个视频建立一个test_dataloader吗？

                with torch.no_grad():
                    # TODO：1,2
                    out_middle_frame = model1_middle_frame(sample)  # TODO:1,3,64,64
                    out_forward_backward = model2_forward_backward(sample)  # TODO:1,2
                    # TODO: 3,4
                    out_consecutive = model3_consecutive_intermittent(full_sample)  # TODO:1,2
                    out_distill = model4_distill(full_sample)  # TODO:1,1080

                logits_fwd = out_forward_backward.numpy()
                reconstruction = out_middle_frame.numpy()[0]
                logits_cons = out_consecutive.numpy()
                logits_resnet = out_distill.numpy()
                probs_fwd = utilss.softmax(logits_fwd[0])  # TODO：正常1,0 异常0,1
                probs_con = utilss.softmax(logits_cons[0])
                class_id = int(meta.numpy()[0][-2]) - 1

                err_1, err_2, err_3, err_4 =np.mean(np.abs(label.numpy()[0] - reconstruction)), probs_fwd[1], probs_con[1], np.abs(meta.numpy()[0][-1] - logits_resnet[0, 1000 + class_id])
                errs_1.append(err_1)
                errs_2.append(err_2)
                errs_3.append(err_3)
                errs_4.append(err_4)
                loc_v3_video.append(meta.numpy()[0][:-2])

            errs_1 = utilss.normalize_(errs_1)
            errs_2 = utilss.normalize_(errs_2)
            errs_3 = utilss.normalize_(errs_3)
            errs_4 = utilss.normalize_(errs_4)

            video_features = np.array(errs_1) + np.array(errs_2) + np.array(errs_3) + np.array(errs_4)  # TODO：对应相加
            np.savetxt(video_features_path, video_features)
            np.save(video_loc_v3_path, loc_v3_video)


        #TODO：计算AUC
        utilss.calculate_AUC()


if HyperParameters.mode == 'train':
    if HyperParameters.ObjectDetectionModel == 'yolov5':
        yolov5_object_extraction()
    resize_video_patch()
    extract_resnet50()
    main_fun(mode='training')
else:
    # yolov5_object_extraction()
    main_fun(mode='testing')


#TODO：程序入口
'''要分开写
0、定义超参数  数据集类型；数据集的入口地址随便；数据集处理的出口地址，要在本地目录，也设置成随便吧；
1、提取目标
2、裁剪目标
3、提取特征
4、正式训练
'''