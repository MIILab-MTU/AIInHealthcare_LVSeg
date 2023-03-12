import argparse
import os
#GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
devicess = [0]

import time
import argparse
import numpy as np

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torchvision import transforms
import torch.distributed as dist
import math
import torchio
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import utils
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR
from tools.save_result import save_train_result,save_val_result

def to_2d(sample_3d):
    patch_size = sample_3d['source']['data'].shape[-3:]
    samples_per_volume = len(sample_3d['source']['data'])
    samples_2d = []
    for i in range(samples_per_volume):
        patch = {
            'source': {'data': sample_3d['source']['data'][i,0, :, :, :]},
            'label': {'data': sample_3d['label']['data'][i,0, :, :, :]}
        }
        for j in range(patch_size[-1]):
            patch_2d = {
                'source': {'data': patch['source']['data'][:, :, j]},
                'label': {'data': patch['label']['data'][:, :, j]}
            }
            samples_2d.append(patch_2d)
    return samples_2d

def get_parser():
    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument('--output_dir', type=str, default='./checkpoints', required=False, help='Directory to save checkpoints')
    parser.add_argument('--latest_checkpoint_file', type=str, default='checkpoint_latest.pt', help='Store the latest checkpoint in each epoch')
    parser.add_argument('--result', type=str, default='./result/U-Net/excel', required=False, help='Directory to save result')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--validate_epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    # data
    parser.add_argument('--patch_x', type=int, default=32)
    parser.add_argument('--patch_y', type=int, default=32)
    parser.add_argument('--patch_z', type=int, default=32)
    parser.add_argument('--source_path', type=str, default="./nii/data(28)/endo")
    parser.add_argument('--in_class', type=int, default=1, help="input")
    parser.add_argument('--out_class', type=int, default=1, help="output")
    parser.add_argument('--aug', type=bool, default=False, help="data augmentation")
    parser.add_argument('--scheduer_gamma', type=int, default=0.8, help="scheduer_gamma")
    parser.add_argument('--scheduer_step_size', type=int, default=20, help="scheduer_step_size")
    # cpu
    parser.add_argument('--n_workers', type=int, default=8)
    #Save model
    parser.add_argument(
        '-k',
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    # CV
    parser.add_argument('--cv', type=int, default=2)  # cross validation, CV=5
    parser.add_argument('--cv_max', type=int, default=5)
    parser.add_argument('--l2', type=float, default=0)
    parser.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    parser.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args=get_parser()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.result, exist_ok=True)
    source_path=args.source_path
    data_path=os.path.join(source_path,'data')
    label_path=os.path.join(source_path,'label')
    from models.two_d.unet import Unet
    model = Unet(in_channels=args.in_class, classes=args.out_class+1)
    model = torch.nn.DataParallel(model, device_ids=devicess)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=args.scheduer_step_size, gamma=args.scheduer_gamma)
    model.cuda()
    from torch.nn.modules.loss import CrossEntropyLoss
    criterion_ce = CrossEntropyLoss().cuda()
    from tools.data_splitting import get_split_deterministic
    data_list=os.listdir(data_path)
    train_list,val_list=get_split_deterministic(data_list,fold=args.cv, num_splits=args.cv_max)
    train_data = [os.path.join(data_path, x) for x in train_list]
    train_label = [os.path.join(label_path, x) for x in train_list]
    val_data = [os.path.join(data_path, x) for x in val_list]
    val_label = [os.path.join(label_path, x) for x in val_list]
    from tools.twoD_data_function import MedData_train,MedData_val
    train_dataset = MedData_train(train_data, train_label,args)
    train_loader = DataLoader(train_dataset.queue_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    val_dataset = MedData_val(val_data, val_label,args)
    val_loader = DataLoader(val_dataset.queue_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=False)
    epochs = args.num_epochs
    iteration = 0
    epochs_dice = []
    epochs_loss = []
    val_dice = []
    val_loss = []
    for epoch in tqdm(range(0, epochs + 1)):
        print("epoch:" + str(epoch))
        epoch_loss = 0.0
        epoch += 1
        model.train()
        num_iters = 0
        batch_loss=[]
        batch_dice=[]
        for i, batch in enumerate(tqdm(train_loader)):
            sample_2d = to_2d(batch)
            patch_dice = []
            patch_loss = []
            for j in tqdm(range(len(sample_2d))):
                x = sample_2d[j]['source']['data']
                y = sample_2d[j]['label']['data']
                x = x.unsqueeze(0).unsqueeze(0)
                y = y.unsqueeze(0).unsqueeze(0)
                y_back = torch.zeros_like(y)
                y_back[(y == 0)] = 1
                x = x.type(torch.FloatTensor).cuda()
                y = torch.cat((y_back, y), 1)
                y = y.type(torch.FloatTensor).cuda()
                outputs = model(x)
                labels = outputs.argmax(dim=1)
                loss = criterion_ce(outputs, y.argmax(dim=1))
                patch_loss.append(loss.item())
                num_iters += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iteration += 1
                y_argmax = y.argmax(dim=1)
                dice = metric(y_argmax.cpu(), labels.cpu())
                patch_dice.append(dice.item())

            batch_dice.append(sum(patch_dice) / len(patch_dice))
            batch_loss.append(sum(patch_loss) / len(patch_loss))

        batch_dice_mean = sum(batch_dice) / len(batch_dice)
        batch_loss_mean = sum(batch_loss) / len(batch_loss)
        epochs_dice.append(batch_dice_mean)
        epochs_loss.append(batch_loss_mean)
        train_loss = epoch_loss / len(train_loader)
        print(train_loss)
        save_train_result(args.result,epoch,epochs_dice,epochs_loss)
        # Store latest checkpoint in each epoch
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,

            },
            os.path.join(args.output_dir, args.latest_checkpoint_file),
        )
        scheduler.step()
        with torch.no_grad():
            if (epoch + 1) % args.validate_epoch == 0:
                model.eval()
                model_results = []
                batch_loss = []
                batch_dice = []
                for i, batch in enumerate(val_loader):
                    sample_2d = to_2d(batch)
                    patch_dice = []
                    for j in range(0, len(sample_2d)):
                        x = sample_2d[j]['source']['data']
                        y = sample_2d[j]['label']['data']
                        x = x.unsqueeze(0).unsqueeze(0)
                        y = y.unsqueeze(0).unsqueeze(0)
                        y_back = torch.zeros_like(y)
                        y_back[(y == 0)] = 1
                        x = x.type(torch.FloatTensor).cuda()
                        y = torch.cat((y_back, y), 1)
                        y = y.type(torch.FloatTensor).cuda()
                        outputs = model(x)
                        y_argmax = y.argmax(dim=1)
                        labels = outputs.argmax(dim=1)
                        dice= metric(y_argmax.cpu(),labels.cpu())
                        patch_dice.append(dice)
                    batch_dice.append(np.mean(np.array(patch_dice)))
                val_dice.append(np.array(batch_dice).mean())
                save_val_result(args.result, epoch,val_dice)