import argparse
import os
from scipy import io
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import PIL
import torch
import timm

# python extract_fea_with_timm.py --data ./data/GenericObjectDecoding-v2/images/training --save_dir ./data/GOD-Wiki/visual_feature/ImageNetTraining --model repvgg_b3g4 --resolution 224

parser = argparse.ArgumentParser(description='PyTorch ImageNet Test')
parser.add_argument('-i', '--data', metavar='./data/GenericObjectDecoding-v2/images/training',
                    help='path to dataset')
parser.add_argument('-o', '--save_dir', metavar='./data/GOD-Wiki/visual_feature/ImageNetTraining',
                    help='path to save')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 100) for test')
parser.add_argument('-r', '--resolution', default=224, type=int,
                    metavar='R', help='resolution (default: 224) for test')
parser.add_argument('-m', '--model', default='resnet50', type=str,
                    metavar='M', help='pretrained model for test')

args = parser.parse_args()
root_dir = args.save_dir+'/pytorch/'+ args.model +'/'
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

def get_default_val_trans(args):
    if (not hasattr(args, 'resolution')) or args.resolution == 224:
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    else:
        trans = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ])
    return trans

def get_ImageNet_val_dataset(args, trans):
    val_dataset = datasets.ImageFolder(args.data, trans)
    return val_dataset

def get_default_ImageNet_val_loader_withpath(args):
    val_trans = get_default_val_trans(args)
    val_dataset = get_ImageNet_val_dataset(args, val_trans)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return val_loader, val_dataset


def extract(val_loader, val_dataset, model_final, model_linear, model_multiscale, use_gpu):
    def save_feature(feat, flag):
        feature_name = feature
        l = feature_name.split('_')
        if 'out' in l:
            l.remove('out')
        if 'list' in l:
            l.remove('list')
        feature_name = '_'.join(l)

        feat = feat.cpu().numpy()
        if flag == 'list':
            dir1 = '{}/{}_{}'.format(root_dir, feature_name, i)
        else:
            dir1 = '{}/{}'.format(root_dir, feature_name)
        if not os.path.exists(dir1):
            os.makedirs(dir1)
        filename = '{}.mat'.format(imid)
        io.savemat(dir1 + '/' + filename, {'feat': feat})

    # switch to evaluate mode
    model_final.eval()
    model_linear.eval()
    model_multiscale.eval()
    # 对应文件夹的label
    # print(val_dataset.class_to_idx)
    with torch.no_grad():
        for i, images in enumerate(val_loader):
            if use_gpu:
                images = images[0].cuda(non_blocking=True)
            final = model_final(images)
            print(f'Original shape: {final.shape}')
            linear = model_linear(images)
            print(f'Pooled shape: {linear.shape}')
            Conv = model_multiscale(images)
            # Conv = [Conv[-4],Conv[-3],Conv[-2],Conv[-1]]
            for x in Conv:
                print(x.shape)

            wnid = val_dataset.imgs[i][0].split("/")[-2]
            imid = val_dataset.imgs[i][0].split("/")[-1].split('.')[0]
            print(wnid, imid)
            feature_list = ['final','linear','Conv']

            for feature in feature_list:
                feat = eval(feature)
                if type(feat) == list:
                    for i in range(len(feat)):
                        save_feature(feat[i], 'list')
                else:
                    save_feature(feat, 'single')

def inference():
    model_final = timm.create_model(args.model, pretrained=True)
    model_linear = timm.create_model(args.model, pretrained=True, num_classes=0)
    model_multiscale = timm.create_model(args.model, pretrained=True, features_only=True)
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
        use_gpu = False
    else:
        model_final = model_final.cuda()
        model_linear = model_linear.cuda()
        model_multiscale = model_multiscale.cuda()
        use_gpu = True

    cudnn.benchmark = True

    val_loader, val_dataset = get_default_ImageNet_val_loader_withpath(args)

    extract(val_loader, val_dataset, model_final, model_linear, model_multiscale, use_gpu)

def extract_no_conv(val_loader, val_dataset, model_final, model_linear, use_gpu):
    def save_feature(feat, flag):
        feature_name = feature
        l = feature_name.split('_')
        if 'out' in l:
            l.remove('out')
        if 'list' in l:
            l.remove('list')
        feature_name = '_'.join(l)

        feat = feat.cpu().numpy()
        if flag == 'list':
            dir1 = '{}/{}_{}'.format(root_dir, feature_name, i)
        else:
            dir1 = '{}/{}'.format(root_dir, feature_name)
        if not os.path.exists(dir1):
            os.makedirs(dir1)
        filename = '{}.mat'.format(imid)
        io.savemat(dir1 + '/' + filename, {'feat': feat})

    # switch to evaluate mode
    model_final.eval()
    model_linear.eval()
    with torch.no_grad():
        for i, images in enumerate(val_loader):
            if use_gpu:
                images = images[0].cuda(non_blocking=True)
            final = model_final(images)
            print(f'Original shape: {final.shape}')
            linear = model_linear(images)
            print(f'Pooled shape: {linear.shape}')

            wnid = val_dataset.imgs[i][0].split("/")[-2]
            imid = val_dataset.imgs[i][0].split("/")[-1].split('.')[0]
            print(wnid, imid)
            feature_list = ['final','linear']

            for feature in feature_list:
                feat = eval(feature)
                if type(feat) == list:
                    for i in range(len(feat)):
                        save_feature(feat[i], 'list')
                else:
                    save_feature(feat, 'single')

def inference_no_conv():
    model_final = timm.create_model(args.model, pretrained=True)
    model_linear = timm.create_model(args.model, pretrained=True, num_classes=0)
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
        use_gpu = False
    else:
        model_final = model_final.cuda()
        model_linear = model_linear.cuda()
        use_gpu = True

    cudnn.benchmark = True

    val_loader, val_dataset = get_default_ImageNet_val_loader_withpath(args)

    extract_no_conv(val_loader, val_dataset, model_final, model_linear, use_gpu)

if __name__ == '__main__':
    inference()
    # inference_no_conv()

