from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.manifold import TSNE
import pandas as pd
import argparse
import os
import shutil
import time
import random
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import wideresnetwithABC as models
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from scipy import optimize
from utils import ramps
parser = argparse.ArgumentParser(description='PyTorch fixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Method options
parser.add_argument('--num_max', type=int, default=1500,
                        help='Number of samples in the maximal class')
parser.add_argument('--label_ratio', type=float, default=20, help='percentage of labeled data')
parser.add_argument('--imb_ratio', type=int, default=100, help='Imbalance ratio')
parser.add_argument('--step', action='store_true', help='Type of class-imbalance')
parser.add_argument('--val-iteration', type=int, default=500,
                        help='Frequency for the evaluation')
parser.add_argument('--num_val', type=int, default=10,
                        help='Number of validation data')
# Hyperparameters for FixMatch
parser.add_argument('--tau', default=0.95, type=float, help='hyper-parameter for pseudo-label of FixMatch')
parser.add_argument('--ema-decay', default=0.999, type=float)
#dataset and imbalanced type
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset')
parser.add_argument('--imbalancetype', type=str, default='long', help='Long tailed or step imbalanced')
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
if args.dataset=='cifar10':
    import dataset.fix_cifar10 as dataset
    print(f'==> Preparing imbalanced CIFAR10')
    num_class = 10
elif args.dataset=='svhn':
    import dataset.fix_svhn as dataset
    print(f'==> Preparing imbalanced SVHN')
    num_class = 10
elif args.dataset=='cifar100':
    import dataset.fix_cifar100 as dataset
    print(f'==> Preparing imbalanced CIFAR100')
    num_class = 100
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
# np.random.seed(args.manualSeed)
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def make_imb_data(max_num, class_num, gamma,imb):
    if imb == 'long':
        mu = np.power(1/gamma, 1/(class_num - 1))
        class_num_list = []
        for i in range(class_num):
            if i == (class_num - 1):
                class_num_list.append(int(max_num / gamma))
            else:
                class_num_list.append(int(max_num * np.power(mu, i)))
        print(class_num_list)
    if imb=='step':
        class_num_list = []
        for i in range(class_num):
            if i < int((class_num) / 2):
                class_num_list.append(int(max_num))
            else:
                class_num_list.append(int(max_num / gamma))
        print(class_num_list)
    return list(class_num_list)
def validate(valloader, model,criterion, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    tsne_data = torch.tensor([])
    tsne_label = torch.tensor([])

    # switch to evaluate mode
    model.eval()

    accperclass = np.zeros((num_class))

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    output_list = []
    label_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(valloader):

            # measure data loading time
            data_time.update(time.time() - end)
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            targetsonehot = torch.zeros(inputs.size()[0], num_class).scatter_(1, targets.cpu().view(-1, 1).long(), 1)
            q=model(inputs)
            outputs2 = model.classify2(q)

            unbiasedscore = F.softmax(outputs2)

            # tsne_data = inputs.view(-1)
            unbiased = torch.argmax(unbiasedscore,axis=1)

            output_list.append(outputs2)
            label_list.append(unbiased)

            # tsne_data = torch.stack([tsne_data, data],dim=1)
            # tsne_label = torch.stack([tsne_label, unbiasedscore],dim=1)

            outputs2onehot = torch.zeros(inputs.size()[0], num_class).scatter_(1, unbiased.cpu().view(-1, 1).long(), 1)
            loss = criterion(outputs2, targets)
            accperclass = accperclass + torch.sum(targetsonehot * outputs2onehot, dim=0).cpu().detach().numpy().astype(np.int64)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs2, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

             # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                          'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
    outputs_all = torch.cat(output_list)
    label_all = torch.cat(label_list)
    if args.dataset=='cifar10':
        accperclass=accperclass/1000
    elif args.dataset=='svhn':
        accperclass=accperclass/1500
    elif args.dataset=='cifar100':
        accperclass=accperclass/100
    return (losses.avg, top1.avg, accperclass,outputs_all,label_all)

N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, num_class, args.imb_ratio,args.imbalancetype)
U_SAMPLES_PER_CLASS = make_imb_data((100-args.label_ratio)/args.label_ratio * args.num_max, num_class, args.imb_ratio,args.imbalancetype)
ir2=N_SAMPLES_PER_CLASS[-1]/np.array(N_SAMPLES_PER_CLASS)
if args.dataset == 'cifar10':
        train_labeled_set, train_unlabeled_set,test_set = dataset.get_cifar10('./data', N_SAMPLES_PER_CLASS,U_SAMPLES_PER_CLASS)
elif args.dataset == 'svhn':
        train_labeled_set, train_unlabeled_set, test_set = dataset.get_SVHN('./data', N_SAMPLES_PER_CLASS,U_SAMPLES_PER_CLASS)
elif args.dataset =='cifar100':
        train_labeled_set, train_unlabeled_set, test_set = dataset.get_cifar100('./data', N_SAMPLES_PER_CLASS,U_SAMPLES_PER_CLASS)

def create_model(ema=False,has_dropout=False):
    model = models.WideResNet(num_classes=num_class,has_dropout = has_dropout)
    model = model.cuda()

    params = list(model.parameters())
    if ema:
        for param in params:
            param.detach_()

    return model, params
ema_model,  _ = create_model(ema=True)


model_weight_path = r'D:\PROJECT\pseudo_label\ABC-main\ABCcode0920\result\2.5loss\model_500.pth.tar'
state_dict = torch.load(model_weight_path)['state_dict']
ema_model.load_state_dict(state_dict)
ahead = 0
temp = []
result = []
true = []
test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
classes = np.array(classes)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax


for batch_idx, (inputs, targets, _) in enumerate(test_loader):

    inputs = inputs.cuda()
    logit = ema_model(inputs)
    outputs2 = ema_model.classify2(logit)
    # output = ema_model(outputs2)
    index,pred_label =torch.topk(outputs2,1)
    for t in targets:
        true.append(t.detach().cpu().item())
    for i in pred_label:
         for j in i:
             temp.append(j.detach().cpu().item())

# data = torch.load(outputs2)
# labels = torch.load(targets)
criterion = nn.CrossEntropyLoss()
test_loss, test_acc, testclassacc,outputs_all,label_all = validate(test_loader, ema_model, criterion, mode='Test Stats ')

# data = torch.load(outputs2)
# labels = torch.load(targets)
criterion = nn.CrossEntropyLoss()
data = outputs_all.cpu().numpy()
labels = label_all.cpu().numpy()

tsne = TSNE(n_components=2, random_state=0, perplexity=15)
embedded_data = tsne.fit_transform(data)
plt.figure(figsize=(10, 10))
plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=labels, cmap=plt.cm.tab10)
plt.axis('off')
plt.colorbar()
plt.show()



S_X1 = np.random.random((200, 200))  # 原始特征

class_num = 10
#####生成标签
y_s = np.zeros((20 * class_num))
for i in range(20 * class_num):
    y_s[i] = i // 20
###################

maker = ['o', 'v', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']  # 设置散点形状
colors = ['black', 'tomato', 'yellow', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
          'hotpink']  # 设置散点颜色

Label_Com = ['S-1', 'T-1', 'S-2', 'T-2', 'S-3',
             'T-3', 'S-4', 'T-4', 'S-5', 'T-5', 'S-6', 'T-6', 'S-7', 'T-7', 'S-8', 'T-8', 'S-9', 'T-9',
             'S-10', 'T-10', 'S-11', 'T-11', 'S-12', 'T-12']  ##图例名称

### 设置字体格式
font1 = {'family': 'Times New Roman',

         'weight': 'bold',
         'size': 32,
         }


def visual(X):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    # '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    return X_norm


def plot_with_labels(S_lowDWeights, Trure_labels, name):
    plt.cla()  # 清除当前图形中的当前活动轴,所以可以重复利用

    # 降到二维了，分别给x和y
    True_labels = Trure_labels.reshape((-1, 1))

    S_data = np.hstack((S_lowDWeights, True_labels))
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})

    for index in range(class_num):
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=100, marker=maker[0], c='red', edgecolors=colors[index], alpha=0.65)

    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    #
    plt.title(name, fontsize=32, fontweight='normal', pad=20)


fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(111)
# plot_with_labels(visual(S_X1), y_s, '(a)')
outputs2 = outputs2.detach().cpu().numpy()
temp = np.array(outputs2)
true = np.array(true)
# plot_with_labels(visual(temp), true, '(a)')

plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None,
                    wspace=0.1, hspace=0.15)
plt.legend(scatterpoints=1, labels=Label_Com, loc='best', labelspacing=0.4, columnspacing=0.4, markerscale=2,
           bbox_to_anchor=(0.9, 0), ncol=12, prop=font1, handletextpad=0.1)

# plt.savefig('./'+str(sour)+str(tar)+'.png', format='png',dpi=300, bbox_inches='tight')
plt.show()
