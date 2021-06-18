import os
import torch
import torch.nn as nn
from torchvision import transforms
import argparse
from models import ResNet
from models import MobileNetV2
from models import MobileNetV2_2
from collections import OrderedDict
from PIL import Image
import time


def use_encoder(args):
    # prepare model
    print('the encoder model is:', args.model)
    if args.model == 'resnet18':
        encoder = ResNet()
    else:
        encoder = MobileNetV2_2()

    print('loading pretrained model...')
    seg_model_checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))['state_dict']
    new_state_dict = OrderedDict()
    for k, v in seg_model_checkpoint.items():
        if 'backbone' in k:
            name = k[16:]
            new_state_dict[name] = v
    checkpoint = new_state_dict

    encoder.load_state_dict(checkpoint)
    # encoder.cuda()

    print('use the model...')
    if args.model_parallel:
        encoder = nn.DataParallel(encoder)
    encoder.eval()

    # use it
    img_file = args.image_path
    to_tensor = transforms.ToTensor()
    image = to_tensor(Image.open(img_file).convert('RGB')).unsqueeze(0)
    if image == None:
        # image = torch.rand(1, 3, 256, 512).cuda()
        image = torch.rand(1,3,256,512)
    else:
        image = image
    time1 = time.time()
    output, _ = encoder(image)
    time2 = time.time()

    print('time cost:', time2-time1)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--checkpoint_path', type=str,
                        default='saved/best_model.pth')
    parser.add_argument('--model_parallel', type=bool, default=False)
    parser.add_argument('--model', type=str, default='mobile')
    parser.add_argument('--image_path', type=str, default='images_folder/00000001.bmp')
    args = parser.parse_args()

    if len(args.gpu.split(',')) > 1:
        args.model_parallel = True

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

    print(use_encoder(args=args))
