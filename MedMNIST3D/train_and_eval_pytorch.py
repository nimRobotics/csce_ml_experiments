import os
import argparse
from tqdm import trange
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from collections import OrderedDict
from models import ResNet18, ResNet50

from acsconv.converters import ACSConverter, Conv3dConverter, Conv2_5dConverter
from utils import model_to_syncbn, Transform3D
import medmnist
from medmnist import INFO, Evaluator

# clear cache
torch.cuda.empty_cache()
# import libAUC
from libauc.losses import AUCMLoss, CrossEntropyLoss, AUCM_MultiLabel
from libauc.optimizers import PESG, Adam
import random
import PIL

def set_random_seeds(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def main(data_flag, output_root, num_epochs, gpu_ids, batch_size, conv, pretrained_3d, download, model_flag, as_rgb, shape_transform, model_path, run, test_flag, libauc_loss, optimizer_type, rotation, scaling, translation):

    lr = 0.001
    gamma=0.1
    milestones = [0.5 * num_epochs, 0.75 * num_epochs]

    info = INFO[data_flag]
    task = info['task']
    n_channels = 3 if as_rgb else info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])
    
    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_ids[0])

    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu') 
    
        
    output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    print('==> Preparing data...')

    # add rotation and scaling
    train_transforms = []
    eval_transforms = []

    # if shape_transform:
    #     train_transforms.append(Transform3D(mul='random'))
    #     eval_transforms.append(Transform3D(mul='0.5'))
    # else:
    #     train_transforms.append(Transform3D())
    #     eval_transforms.append(Transform3D())

    # # add to tensor
    # train_transforms.append(transforms.ToTensor())
    # eval_transforms.append(transforms.ToTensor())

    # if rotation is not None:
    #     print('==> Randomly rotate the images by {} degrees...'.format(rotation))
    #     train_transforms.append(transforms.RandomRotation(rotation))
    #     eval_transforms.append(transforms.RandomRotation(rotation))

    # if scaling:
    #     print('==> Randomly scale the images by 0.9 to 1.1...')
    #     train_transforms.append(transforms.RandomResizedCrop(28, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=PIL.Image.NEAREST))
    #     eval_transforms.append(transforms.RandomResizedCrop(28, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=PIL.Image.NEAREST))

    # if translation:
    #     print('==> Randomly translate the images by 0.1...')
    #     train_transforms.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))
    #     eval_transforms.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))

    

    # train_transform = transforms.Compose(train_transforms)
    # eval_transform = transforms.Compose(eval_transforms)

    # train_transform = Transform3D(mul='random') if shape_transform else Transform3D()
    # eval_transform = Transform3D(mul='0.5') if shape_transform else Transform3D()

    train_transform = Transform3D(mul='random', rotation=rotation, scale=scaling, translate=translation) if shape_transform else Transform3D()
    eval_transform = Transform3D(mul='0.5', rotation=rotation, scale=scaling, translate=translation) if shape_transform else Transform3D()


     
    train_dataset = DataClass(split='train', transform=train_transform, download=download, as_rgb=as_rgb)
    train_dataset_at_eval = DataClass(split='train', transform=eval_transform, download=download, as_rgb=as_rgb)
    val_dataset = DataClass(split='val', transform=eval_transform, download=download, as_rgb=as_rgb)
    test_dataset = DataClass(split='test', transform=eval_transform, download=download, as_rgb=as_rgb)

    
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset_at_eval,
                                batch_size=batch_size,
                                shuffle=False)
    val_loader = data.DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False)

    print('==> Building and training model...')

    if model_flag == 'resnet18':
        model = ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif model_flag == 'resnet50':
        model = ResNet50(in_channels=n_channels, num_classes=n_classes)
    else:
        raise NotImplementedError

    if conv=='ACSConv':
        model = model_to_syncbn(ACSConverter(model))
    if conv=='Conv2_5d':
        model = model_to_syncbn(Conv2_5dConverter(model))
    if conv=='Conv3d':
        if pretrained_3d == 'i3d':
            model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=-3))
        else:
            model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=None))
    
    model = model.to(device)

    train_evaluator = medmnist.Evaluator(data_flag, 'train')
    val_evaluator = medmnist.Evaluator(data_flag, 'val')
    test_evaluator = medmnist.Evaluator(data_flag, 'test')

    if libauc_loss:
        criterion = AUCMLoss()
        print('Using AUCMLoss')
    else:
        criterion = nn.CrossEntropyLoss()

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=True)
        train_metrics = test(model, train_evaluator, train_loader_at_eval, criterion, device, run, output_root)
        val_metrics = test(model, val_evaluator, val_loader, criterion, device, run, output_root)
        test_metrics = test(model, test_evaluator, test_loader, criterion, device, run, output_root)

        print('train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2]) + \
              'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2]) + \
              'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2]))

    if num_epochs == 0:
        return

    if optimizer_type == 'adam':
        print('Using Adam optimizer')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'sgd':
        print('Using SGD optimizer')
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    elif optimizer_type == 'pesg':
        print('Using PESG optimizer')
        optimizer = PESG(model, loss_fn=criterion, lr=lr, momentum=0.9, weight_decay=1e-4)


    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    logs = ['loss', 'auc', 'acc']
    train_logs = ['train_'+log for log in logs]
    val_logs = ['val_'+log for log in logs]
    test_logs = ['test_'+log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs+val_logs+test_logs, 0)
    
    writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results'))

    best_auc = 0
    best_epoch = 0
    best_model = model

    global iteration
    iteration = 0

    for epoch in trange(num_epochs):
        
        train_loss = train(model, train_loader, criterion, optimizer, device, writer)
        
        train_metrics = test(model, train_evaluator, train_loader_at_eval, criterion, device, run)
        val_metrics = test(model, val_evaluator, val_loader, criterion, device, run)
        test_metrics = test(model, test_evaluator, test_loader, criterion, device, run)
        
        scheduler.step()
        
        for i, key in enumerate(train_logs):
            log_dict[key] = train_metrics[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_metrics[i]

        for key, value in log_dict.items():
            writer.add_scalar(key, value, epoch)
            
        cur_auc = val_metrics[1]
        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_model = model

            print('cur_best_auc:', best_auc)
            print('cur_best_epoch', best_epoch)

    state = {
        'net': model.state_dict(),
    }

    path = os.path.join(output_root, 'best_model.pth')
    torch.save(state, path)

    train_metrics = test(best_model, train_evaluator, train_loader_at_eval, criterion, device, run, output_root)
    val_metrics = test(best_model, val_evaluator, val_loader, criterion, device, run, output_root)
    train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
    val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])

    if test_flag:
        test_metrics = test(best_model, test_evaluator, test_loader, criterion, device, run, output_root)
        test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])
    else:
        test_log = ''

    log = '%s\n' % (data_flag) + train_log + val_log + test_log + \
            'task: %s \n' % (task) + \
            'batch_size: %d  lr: %f  num_epochs: %d  optimizer: %s  optimizer_type: %s  milestones: %s  gamma: %f\n' % \
            (batch_size, lr, num_epochs, optimizer, optimizer_type, milestones, gamma) + \
            'rotation: %d  translation: %d  scale: %d  libauc_loss: %d  \n' % \
            (rotation, translation, scale, libauc_loss) + \
            'conv type: %s  \n' % (conv) 
    print(log)
    
    with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
        f.write(log)        
            
    writer.close()


def train(model, train_loader, criterion, optimizer, device, writer):
    total_loss = []
    global iteration

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        targets = torch.squeeze(targets, 1).long().to(device)
        loss = criterion(outputs, targets)

        total_loss.append(loss.item())
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        iteration += 1

        loss.backward()
        optimizer.step()
    
    epoch_loss = sum(total_loss)/len(total_loss)
    return epoch_loss


def test(model, evaluator, data_loader, criterion, device, run, save_folder=None):

    model.eval()

    total_loss = []
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))
        
            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)
            m = nn.Softmax(dim=1)
            outputs = m(outputs).to(device)
            targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())

            y_score = torch.cat((y_score, outputs), 0)

        y_score = y_score.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score, save_folder, run)

        test_loss = sum(total_loss) / len(total_loss)

        return [test_loss, auc, acc]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST3D')

    parser.add_argument('--data_flag',
                        default='organmnist3d',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models',
                        type=str)
    parser.add_argument('--num_epochs',
                        default=100,
                        help='num of epochs of training, the script would only test model if set num_epochs to 0',
                        type=int)
    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)
    parser.add_argument('--batch_size',
                        default=32,
                        type=int)
    parser.add_argument('--conv',
                        default='ACSConv',
                        help='choose converter from Conv2_5d, Conv3d, ACSConv',
                        type=str)
    parser.add_argument('--pretrained_3d',
                        default='i3d',
                        type=str)
    parser.add_argument('--download',
                        action="store_true")
    parser.add_argument('--as_rgb',
                        help='to copy channels, tranform shape 1x28x28x28 to 3x28x28x28',
                        action="store_true")
    parser.add_argument('--shape_transform',
                        help='for shape dataset, whether multiply 0.5 at eval',
                        action="store_true")
    parser.add_argument('--model_path',
                        default=None,
                        help='root of the pretrained model to test',
                        type=str)
    parser.add_argument('--model_flag',
                        default='resnet18',
                        help='choose backbone, resnet18/resnet50',
                        type=str)
    parser.add_argument('--run',
                        default='model1',
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)
    parser.add_argument('--test_flag',
                        action="store_true")
    parser.add_argument('--libauc_loss',
                        action="store_true")
    parser.add_argument('--optimizer',
                        default='adam',
                        help='choose optimizer from adam, sgd, pesg',
                        type=str)
    parser.add_argument('--rotation',
                        default=None,
                        help='rotation angle of data augmentation',
                        type=int)
    parser.add_argument('--scale',
                        action="store_true",
                        help='scaling for data augmentation')
    parser.add_argument('--translation',
                        action="store_true",
                        help='translation for data augmentation')


    args = parser.parse_args()
    data_flag = args.data_flag
    output_root = args.output_root
    num_epochs = args.num_epochs
    gpu_ids = args.gpu_ids
    batch_size = args.batch_size
    conv = args.conv
    pretrained_3d = args.pretrained_3d
    download = args.download
    model_flag = args.model_flag
    as_rgb = args.as_rgb
    model_path = args.model_path
    shape_transform = args.shape_transform
    run = args.run
    test_flag = args.test_flag
    libauc_loss = args.libauc_loss
    optimizer_type = args.optimizer
    rotation = args.rotation
    scale = args.scale
    translation = args.translation

    main(data_flag, 
        output_root, 
        num_epochs, 
        gpu_ids, 
        batch_size, 
        conv, 
        pretrained_3d, 
        download, 
        model_flag, 
        as_rgb, 
        shape_transform, 
        model_path, 
        run,
        test_flag,
        libauc_loss,
        optimizer_type,
        rotation,
        scale,
        translation)
