from __future__ import print_function
import argparse
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, LLCMData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from model import embed_net
from utils import *
import pdb
import scipy.io

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline: resnet50')
parser.add_argument('--resume', '-r', default='ss1.t', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--save_epoch', default=20, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str, help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=192, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=16, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=16, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='awg', type=str, metavar='m', help='method type: base or awg')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int, help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--gpu', default='1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor for sysu')
parser.add_argument('--tvsearch', action='store_true', help='whether thermal to visible search on RegDB')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dataset = args.dataset
if dataset == 'sysu':
    data_path = '/data/data/sysu/'
    n_class = 395
    test_mode = [1, 2]
    dim = 2048
elif dataset =='regdb':
    data_path = '/data/data/regdb/RegDB/'
    n_class = 206
    test_mode = [2, 1]
    dim = 1024
elif dataset =='llcm':
    data_path = '/media/data2/zyk/LLCM/AEEN/KaiYuan/Datasets/LLCM_V1/'
    n_class = 713
    test_mode = [2, 1] #[2, 1]: VIS to IR; [1, 2]: IR to VIS
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0 
pool_dim = 1024 * 4
print('==> Building model..')
if args.method =='base':
    net = embed_net(n_class, dataset, no_local= 'off', gm_pool =  'off', arch=args.arch)
else:
    net = embed_net(n_class, dataset, no_local= 'on', gm_pool = 'on', arch=args.arch)
#net = nn.DataParallel(net)
net.to(device)    
cudnn.benchmark = True

checkpoint_path = args.model_path

if args.method =='id':
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((args.img_h,args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h,args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def extract_gall_feat(gall_loader):
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat1 = np.zeros((ngall, dim * 4))
    gall_feat2 = np.zeros((ngall, dim * 4))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input1 = Variable(input.cuda())
            input2 = Variable(fliplr(input).cuda())
            feat_pool1, feat_fc1 = net(input1, input1, test_mode[0])
            feat_pool2, feat_fc2 = net(input2, input2, test_mode[0])
            feat = feat_pool1 + feat_pool2
            feat_att = feat_fc1 + feat_fc2
            gall_feat1[ptr:ptr + batch_num, :] = feat[:batch_num].detach().cpu().numpy()
            gall_feat2[ptr:ptr + batch_num, :] = feat_att[:batch_num].detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat1, gall_feat2
    
def extract_query_feat(query_loader):
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat1 = np.zeros((nquery, dim * 4))
    query_feat2 = np.zeros((nquery, dim * 4))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input1 = Variable(input.cuda())
            input2 = Variable(fliplr(input).cuda())
            feat_pool1, feat_fc1 = net(input1, input1, test_mode[1])
            feat_pool2, feat_fc2 = net(input2, input2, test_mode[1])
            feat = feat_pool1 + feat_pool2
            feat_att = feat_fc1 + feat_fc2
            query_feat1[ptr:ptr + batch_num, :] = feat[:batch_num].detach().cpu().numpy()
            query_feat2[ptr:ptr + batch_num, :] = feat_att[:batch_num].detach().cpu().numpy()
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat1, query_feat2


if dataset == 'sysu':

    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = checkpoint_path + args.resume
        # model_path = checkpoint_path + 'sysu_awg_p4_n8_lr_0.1_seed_0_best.t'
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat1, query_feat2 = extract_query_feat(query_loader)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=trial)

        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat1, gall_feat2 = extract_gall_feat(trial_gall_loader)

        # fc feature
        distmat1 = np.matmul(query_feat1, np.transpose(gall_feat1))
        distmat2 = np.matmul(query_feat2, np.transpose(gall_feat2))
        distmat3 = distmat1 + distmat2
        a = 0.2
        distmat7 = (1 - a) * distmat1 + a* distmat2
        
        cmc1, mAP1, mINP1 = eval_sysu(-distmat1, query_label, gall_label, query_cam, gall_cam)
        cmc2, mAP2, mINP2 = eval_sysu(-distmat2, query_label, gall_label, query_cam, gall_cam)
        cmc3, mAP3, mINP3 = eval_sysu(-distmat3, query_label, gall_label, query_cam, gall_cam)
        cmc7, mAP7, mINP7 = eval_sysu(-distmat7, query_label, gall_label, query_cam, gall_cam)


        if trial == 0:

            all_cmc1 = cmc1
            all_mAP1 = mAP1
            all_mINP1 = mINP1

            all_cmc2 = cmc2
            all_mAP2 = mAP2
            all_mINP2 = mINP2

            all_cmc3 = cmc3
            all_mAP3 = mAP3
            all_mINP3 = mINP3

            all_cmc7 = cmc7
            all_mAP7 = mAP7
            all_mINP7 = mINP7

        else:
            all_cmc1 = all_cmc1 + cmc1
            all_mAP1 = all_mAP1 + mAP1
            all_mINP1 = all_mINP1 + mINP1

            all_cmc2 = all_cmc2 + cmc2
            all_mAP2 = all_mAP2 + mAP2
            all_mINP2 = all_mINP2 + mINP2

            all_cmc3 = all_cmc3 + cmc3
            all_mAP3 = all_mAP3 + mAP3
            all_mINP3 = all_mINP3 + mINP3

            all_cmc7 = all_cmc7 + cmc7
            all_mAP7 = all_mAP7 + mAP7
            all_mINP7 = all_mINP7 + mINP7


        print('Test Trial: {}'.format(trial))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc1[0], cmc1[4], cmc1[9], cmc1[19], mAP1, mINP1))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc2[0], cmc2[4], cmc2[9], cmc2[19], mAP2, mINP2))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc3[0], cmc3[4], cmc3[9], cmc3[19], mAP3, mINP3))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc7[0], cmc7[4], cmc7[9], cmc7[19], mAP7, mINP7))

elif dataset == 'llcm':

    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = checkpoint_path + args.resume
        # model_path = checkpoint_path + 'llcm_agw_p4_n8_lr_0.1_seed_0_best.t'
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # testing set
    query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
    gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=0)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat1, query_feat2 = extract_query_feat(query_loader)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=trial)

        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat1, gall_feat2 = extract_gall_feat(trial_gall_loader)

        # fc feature

        distmat1 = np.matmul(query_feat1, np.transpose(gall_feat1))
        distmat2 = np.matmul(query_feat2, np.transpose(gall_feat2))
        distmat3 = distmat1 + distmat2
        a = 0.1
        distmat7 = (1 - a) * distmat1 + a* distmat2
        
        cmc1, mAP1, mINP1 = eval_llcm(-distmat1, query_label, gall_label, query_cam, gall_cam)
        cmc2, mAP2, mINP2 = eval_llcm(-distmat2, query_label, gall_label, query_cam, gall_cam)
        cmc3, mAP3, mINP3 = eval_llcm(-distmat3, query_label, gall_label, query_cam, gall_cam)
        cmc7, mAP7, mINP7 = eval_llcm(-distmat7, query_label, gall_label, query_cam, gall_cam)


        if trial == 0:

            all_cmc1 = cmc1
            all_mAP1 = mAP1
            all_mINP1 = mINP1

            all_cmc2 = cmc2
            all_mAP2 = mAP2
            all_mINP2 = mINP2

            all_cmc3 = cmc3
            all_mAP3 = mAP3
            all_mINP3 = mINP3

            all_cmc7 = cmc7
            all_mAP7 = mAP7
            all_mINP7 = mINP7

        else:
            all_cmc1 = all_cmc1 + cmc1
            all_mAP1 = all_mAP1 + mAP1
            all_mINP1 = all_mINP1 + mINP1

            all_cmc2 = all_cmc2 + cmc2
            all_mAP2 = all_mAP2 + mAP2
            all_mINP2 = all_mINP2 + mINP2

            all_cmc3 = all_cmc3 + cmc3
            all_mAP3 = all_mAP3 + mAP3
            all_mINP3 = all_mINP3 + mINP3

            all_cmc7 = all_cmc7 + cmc7
            all_mAP7 = all_mAP7 + mAP7
            all_mINP7 = all_mINP7 + mINP7


        print('Test Trial: {}'.format(trial))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc1[0], cmc1[4], cmc1[9], cmc1[19], mAP1, mINP1))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc2[0], cmc2[4], cmc2[9], cmc2[19], mAP2, mINP2))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc3[0], cmc3[4], cmc3[9], cmc3[19], mAP3, mINP3))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc7[0], cmc7[4], cmc7[9], cmc7[19], mAP7, mINP7))


elif dataset == 'regdb':

    for trial in range(10):
        test_trial = trial +1
        model_path = checkpoint_path + args.resume
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])

        # training set
        trainset = RegDBData(data_path, test_trial, transform=transform_train)
        # generate the idx of each person identity
        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

        # testing set
        query_img, query_label = process_test_regdb(data_path, trial=test_trial, modal='visible')
        gall_img, gall_label = process_test_regdb(data_path, trial=test_trial, modal='thermal')

        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        nquery = len(query_label)
        ngall = len(gall_label)

        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        print('Data Loading Time:\t {:.3f}'.format(time.time() - end))


        query_feat1, query_feat2 = extract_query_feat(query_loader)
        gall_feat1, gall_feat2 = extract_gall_feat(gall_loader)

        if args.tvsearch:
            # fc feature
            distmat1 = np.matmul(gall_feat1, np.transpose(query_feat1))
            distmat2 = np.matmul(gall_feat2, np.transpose(query_feat2))
            distmat3 = distmat1 + distmat2
            a = 0.0
            distmat7 = (1 - a) * distmat1 + a* distmat2

            cmc1, mAP1, mINP1 = eval_regdb(-distmat1, gall_label, query_label)
            cmc2, mAP2, mINP2 = eval_regdb(-distmat2, gall_label, query_label)
            cmc3, mAP3, mINP3 = eval_regdb(-distmat3, gall_label, query_label)
            cmc7, mAP7, mINP7 = eval_regdb(-distmat7, gall_label, query_label)

        else:
            # fc feature
            distmat1 = np.matmul(query_feat1, np.transpose(gall_feat1))
            distmat2 = np.matmul(query_feat2, np.transpose(gall_feat2))
            distmat3 = distmat1 + distmat2
            a = 0.1
            distmat7 = (1 - a) * distmat1 + a* distmat2

            cmc1, mAP1, mINP1 = eval_regdb(-distmat1, query_label, gall_label)
            cmc2, mAP2, mINP2 = eval_regdb(-distmat2, query_label, gall_label)
            cmc3, mAP3, mINP3 = eval_regdb(-distmat3, query_label, gall_label)
            cmc7, mAP7, mINP7 = eval_regdb(-distmat7, query_label, gall_label)


        if trial == 0:

            all_cmc1 = cmc1
            all_mAP1 = mAP1
            all_mINP1 = mINP1

            all_cmc2 = cmc2
            all_mAP2 = mAP2
            all_mINP2 = mINP2

            all_cmc3 = cmc3
            all_mAP3 = mAP3
            all_mINP3 = mINP3

            all_cmc7 = cmc7
            all_mAP7 = mAP7
            all_mINP7 = mINP7

        else:
            all_cmc1 = all_cmc1 + cmc1
            all_mAP1 = all_mAP1 + mAP1
            all_mINP1 = all_mINP1 + mINP1

            all_cmc2 = all_cmc2 + cmc2
            all_mAP2 = all_mAP2 + mAP2
            all_mINP2 = all_mINP2 + mINP2

            all_cmc3 = all_cmc3 + cmc3
            all_mAP3 = all_mAP3 + mAP3
            all_mINP3 = all_mINP3 + mINP3

            all_cmc7 = all_cmc7 + cmc7
            all_mAP7 = all_mAP7 + mAP7
            all_mINP7 = all_mINP7 + mINP7

        print('Test Trial: {}'.format(trial))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc1[0], cmc1[4], cmc1[9], cmc1[19], mAP1, mINP1))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc2[0], cmc2[4], cmc2[9], cmc2[19], mAP2, mINP2))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc3[0], cmc3[4], cmc3[9], cmc3[19], mAP3, mINP3))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc7[0], cmc7[4], cmc7[9], cmc7[19], mAP7, mINP7))

cmc1 = all_cmc1 / 10
mAP1 = all_mAP1 / 10
mINP1 = all_mINP1 / 10

cmc2 = all_cmc2 / 10
mAP2 = all_mAP2 / 10
mINP2 = all_mINP2 / 10

cmc3 = all_cmc3 / 10
mAP3 = all_mAP3 / 10
mINP3 = all_mINP3 / 10

cmc7 = all_cmc7 / 10
mAP7 = all_mAP7 / 10
mINP7 = all_mINP7 / 10

print('All Average:')
print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc1[0], cmc1[4], cmc1[9], cmc1[19], mAP1, mINP1))
        
print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc2[0], cmc2[4], cmc2[9], cmc2[19], mAP2, mINP2))
        
print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc3[0], cmc3[4], cmc3[9], cmc3[19], mAP3, mINP3))
        
print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc7[0], cmc7[4], cmc7[9], cmc7[19], mAP7, mINP7))

