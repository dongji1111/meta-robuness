from pyt_common import *
from flags_holder import FlagHolder
import torch
from torchvision import datasets,transforms
from loader import StridedImageFolder
import argparse
from meta import Meta
import numpy

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# input parameter:
# FLAGS.dataset, FLAGS.attack, FLAGS.epsilon,F LAGS.n_iters,
# FLAGS.step_size,FLAGS.class_downsample_factor,FLAGS.resnet_size
# FLAGS.batch_size


def attack_gen(**flag_kwargs):
    FLAGS = FlagHolder()
    FLAGS.initialize(**flag_kwargs)
    attack = get_attack(FLAGS.dataset, FLAGS.attack, FLAGS.epsilon,
                    FLAGS.n_iters, FLAGS.step_size, False)
    attack = attack()
    if FLAGS.dataset in ['cifar-10']:
        nb_classes = 10
    else:
        nb_classes = 1000 // FLAGS.class_downsample_factor
    config = [
        ('conv2d', [32, 3, 1, 1, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 1, 1, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 1, 1, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 1, 1, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 288])
    ]
    model = Meta(args, config).net
    #model = get_model(FLAGS.dataset, FLAGS.resnet_size, nb_classes)
# from evaluator
    if FLAGS.dataset in ['cifar-10']:
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        val_dataset = datasets.CIFAR10(
            root='./', download=True, train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize, ]))
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size= FLAGS.batch_size,
            num_workers=8, pin_memory=True, shuffle=False)
    else:
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        val_dataset = StridedImageFolder(
            '/mnt/imagenet-test/',
                transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,]),
                stride=10)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size= FLAGS.batch_size,
            sampler=val_sampler, num_workers=1, pin_memory=True,
            shuffle=False)

    for batch_idx, (data, target) in enumerate(val_loader):
        print(target.size(),target.dtype)
        #print(target)
        if torch.cuda.is_available():
            rand_target = torch.randint(
                0, nb_classes - 1, target.size(),
                dtype=target.dtype, device='cuda')
        else:
            rand_target = torch.randint(
                0, nb_classes - 1, target.size(),
                dtype=target.dtype, device='cpu')
        data_adv = attack(model, data, rand_target, avoid_target=True, scale_eps=False)
        print(data_adv.shape)
        #print(data_adv)

        if batch_idx == 0:
            break


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1000)
    argparser.add_argument('--n_way', type=int, help='n way', default=10)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=20)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    args = argparser.parse_args()

    dic1 = {'dataset': 'cifar-10', 'class_downsample_factor': 100, 'attack': 'pgd_linf','epsilon': 16.0,
            'n_iters': 1, 'step_size': 1, 'resnet_size': 20, 'batch_size': 1}
    attack_gen(**dic1)