import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import argparse
from meta import Meta
import random
from pyt_common import *


def cifar_resnet(num_block):
    'para [ch_out, ch_in, kernelsz, kernelsz, stride, padding]'
    config = [('conv2d', [16, 3, 3, 3, 1, 1]),
              ('bn', [16]),
              ('relu', [True])]
    for i in range(num_block):
        config.append(('basicblock',[16, 16, 3, 3, 1, 1]))
    config.append(('basicblock',[32, 16, 3, 3, 2, 1]))
    for i in range(num_block-1):
        config.append(('basicblock',[32, 32, 3, 3, 1, 1]))
    config.append(('basicblock', [64, 32, 3, 3, 2, 1]))
    for i in range(num_block-1):
        config.append(('basicblock',[64, 64, 3, 3, 1, 1]))
    config.append(('avg_pool2d',[8]))
    config.append(('flatten', []))
    config.append(('linear', [args.n_way, 64]))
    return config


def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    attack_list = []
    attack_name =['pgd_linf', 'pgd_l2', 'fw_l1', 'jpeg_linf', 'jpeg_l1', 'jpeg_l2','elastic', 'fog', 'gabor', 'snow']
    attack_eps_cifar = [[1,2,4,8],[40,80,160,320],[195,390,780,1560],
                  [0.03125,0.0625,0.125,0.25],[2,8,64,256],[0.125,0.25,0.5,1]]
    for i in range(len(attack_eps_cifar)):
        for j in range(len(attack_eps_cifar[i])):
            attack_list.append([attack_name[i], attack_eps_cifar[i][j]])
    random.seed(1111)
    attack_list = random.shuffle(attack_list)
    attack_list_train = attack_list[0:16]
    attack_list_val = attack_list[16::]
    #print(len(attack_list))
    print(args)

    '''config = [
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
    ]'''
    # create resnet_56 (6n+2,n=9)
    config = cifar_resnet(9)
    print(config)
    #device = torch.device('cuda')
    device = torch.device('cpu')
    maml = Meta(args, config).to(device)
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    #print(maml)
    print('Total trainable tensors:', num)
    train_dataset = datasets.CIFAR10('./cifar10',train=True,download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomCrop(32, 4),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
                                     )

    val_dataset = datasets.CIFAR10('./cifar10',train=False,download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
                                     )


    for epoch in range(args.epoch):
        # fetch meta_batchsz num of episode each time
        spt_size = args.n_way*args.k_spt
        qry_size = args.n_way*args.k_qry
        batch_size = spt_size+qry_size
        db = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=1, pin_memory=True)

        for step, (x, y) in enumerate(db):
            x = x.to(device)
            y = y.to(device)
            x_spt, y_spt, x_qry, y_qry = [],[],[],[]
            x_spt_clean, x_qry_clean = x[:50], x[50:200]
            y_spt_clean, y_qry_clean = y[:50], y[50:200]

            for i in range(len(attack_list_train)):
                x_spt_attack, x_qry_attack = x_spt_clean, x_qry_clean
                # attack the picture
                attack = get_attack('cifar-10', attack_list_train[i][0], attack_list_train[i][1], 10, 1, False)
                attack = attack()
                rand_target = torch.randint(0, 9, size=y_spt_clean.size(), device=device)
                x_spt_attack = attack(maml.net, x_spt_clean,rand_target,
                                      avoid_target=True, scale_eps=False)
                rand_target = torch.randint(0, 9, size=y_qry_clean.size(), device=device)
                x_qry_attack = attack(maml.net, x_qry_clean,rand_target,
                                      avoid_target=True, scale_eps=False)
                x_spt_task = torch.cat([x_spt_clean,x_spt_attack], dim=0)
                x_qry_task = torch.cat([x_qry_clean, x_qry_attack], dim=0)
                y_spt_task = torch.cat([y_spt_clean, y_spt_clean], dim=0)
                y_qry_task = torch.cat([y_qry_clean, y_qry_clean], dim=0)
                # add it to the training data
                x_spt.append(x_spt_task.detach().numpy().tolist())
                x_qry.append(x_qry_task.detach().numpy().tolist())
                y_spt.append(y_spt_task.detach().numpy().tolist())
                y_qry.append(y_qry_task.detach().numpy().tolist())
            x_spt, y_spt, x_qry, y_qry = torch.tensor(x_spt).to(device), torch.tensor(y_spt).to(device),\
                                         torch.tensor(x_qry).to(device), torch.tensor(y_qry).to(device)
            accs, acc = maml(x_spt, y_spt, x_qry, y_qry)

            if step % 1 == 0:
                print('step:', step, '\ttraining acc:', acc)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1000)
    argparser.add_argument('--n_way', type=int, help='n way', default=10)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=10)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    args = argparser.parse_args()

    main()
