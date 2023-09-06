import torch
import torch.nn as nn
import torch.nn.functional as F

#TODO：OK！
class Encoder_deep_wide(torch.nn.Module):
    def __init__(self):
        super(Encoder_deep_wide, self).__init__()
        self.inchannels = 3
        self.encoder = nn.Sequential(
            nn.Conv3d(self.inchannels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU()
            # nn.MaxPool3d((6, 2, 2), stride=(2, 2, 2))
        )

    def forward(self, x):
        x = self.encoder(x)
        x = nn.MaxPool3d((x.shape[2], 2, 2), stride=(2, 2, 2))(x)#TODO：压缩时间维度
        x = x.view(-1, 64, 4, 4)
        return x

#TODO：OK！
class MiddleFrameDecoder(torch.nn.Module):
    def __init__(self):
        super(MiddleFrameDecoder, self).__init__()
        self.s1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.s2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.s3 = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )

    def forward(self, x):
        x = F.interpolate(input=x, size=(x.shape[2] * 2, x.shape[3] * 2))
        x = self.s1(x)
        x = F.interpolate(input=x, size=(x.shape[2] * 2, x.shape[3] * 2))
        x = self.s2(x)
        x = F.interpolate(input=x, size=(x.shape[2] * 2, x.shape[3] * 2))
        x = self.s2(x)
        x = F.interpolate(input=x, size=(x.shape[2] * 2, x.shape[3] * 2))
        x = self.s3(x)
        return x

#TODO：OK！   11111111111111111111111111111111111111111111111111
class MiddleFrame(torch.nn.Module):
    def __init__(self):
        super(MiddleFrame, self).__init__()
        self.encoder_deep_wide = Encoder_deep_wide()
        self.decoder = MiddleFrameDecoder()

    def forward(self, x):
        x = self.encoder_deep_wide(x)#TODO：第一个阶段
        x = self.decoder(x)
        return x



class DecoderForwardBackward(torch.nn.Module):
    def __init__(self):
        super(DecoderForwardBackward, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        x = self.decoder(x)
        return x

#TODO：OK!  222222222222222222222222
class ForwardBackward(torch.nn.Module):
    def __init__(self):
        super(ForwardBackward, self).__init__()
        self.encoder_deep_wide = Encoder_deep_wide()
        self.decoder_forward_backward = DecoderForwardBackward()

    def forward(self, x):
        x = self.encoder_deep_wide(x)  # TODO：第一个阶段
        x = self.decoder_forward_backward(x)
        return x



class DecoderConsecutiveIntermittent(torch.nn.Module):
    def __init__(self):
        super(DecoderConsecutiveIntermittent, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

#TODO：OK！  333333333333333333333333333333333333
class ConsecutiveIntermittent(torch.nn.Module):
    def __init__(self):
        super(ConsecutiveIntermittent, self).__init__()
        self.encoder_deep_wide = Encoder_deep_wide()
        self.decoder_consecutive_intermittent = DecoderConsecutiveIntermittent()

    def forward(self, x):
        x = self.encoder_deep_wide(x)
        x = self.decoder_consecutive_intermittent(x)
        return x



class DecoderDistill(torch.nn.Module):
    def __init__(self):
        super(DecoderDistill, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.Linear(128, 1080)
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

#TODO：OK！  4444444444444444444444444444444444
class Distill(torch.nn.Module):
    def __init__(self):
        super(Distill, self).__init__()
        self.encoder_deep_wide = Encoder_deep_wide()
        self.decoder_distill = DecoderDistill()

    def forward(self, x):
        x = self.encoder_deep_wide(x)
        x = self.decoder_distill(x)
        return x


# #TODO：定义数据
# input = torch.randn([8,3,6,64,64])
#
# #TODO：定义模型
# model = Distill()
#
# #TODO：测试模型
# output = model(input)
# print(output)
# print(output.shape)



