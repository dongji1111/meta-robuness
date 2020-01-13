import torch
import  numpy as np
from torchvision import datasets, transforms, models
from    torch.utils.data import DataLoader
import  argparse
from meta import Meta
'''datasetlist = ['MNIST','imagenet']
attacklist1= ['Loo_1', 'Loo_2', 'Loo_4', 'Loo_8',
              'L2_40', 'L2_80', 'L2_160', 'L2_320',
              'L1_195', 'L1_390', 'L1_780', 'L1_1560',
              'Loo_JPEG_0.03125', 'Loo_JPEG_0.0625', 'Loo_JPEG_0.125', 'Loo_JPEG_0.25',
              'L1_JPEG_2', 'L1_JPEG_8', 'L1_JPEG_64', 'Loo_JPEG_256',
              'Elastic_0.125', 'Elastic_0.25', 'Elastic_0.5', 'Elastic_1']
attacklist2 = ['Loo_1', 'Loo_2', 'Loo_4',
               'L2_150', 'L2_300', 'L2_600',
               'L1_9562.5', 'L1_19125', 'L1_76500',
               'Loo_JPEG_0.03125', 'Loo_JPEG_0.0625', 'Loo_JPEG_0.125',
                'L2_JPEG_8', 'L2_JPEG_16', 'L2_JPEG_32',
               'L1_JPEG_256', 'L1_JPEG_1024', 'L1_JPEG_4096',
               'Elastic_0.25', 'Elastic_0.5', 'Elastic_2',
               'Fog_128','Fog_256','Fog_512',
               'Gabor_6.25','Gabor_12.5','Gabor_25',
               'Snow_0.0625','Snow_0.125','Snow_0.25',]

attacklist_all = []

for attack in attacklist1:
    new_attack = datasetlist[0]+'_'+attack
    attacklist_all.append(new_attack)
for attack in attacklist2:
    new_attack = datasetlist[1]+'_'+attack
    attacklist_all.append(new_attack)

for i in range(6):
    print(attacklist_all[random.randint(0,18)])
'''
def train():
    train_dataset = datasets.CIFAR10('./cifar10', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomCrop(32, 4),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
                                 )
    db = DataLoader(train_dataset, 200, shuffle=True, num_workers=1, pin_memory=True)
    for step, (x_spt, y_spt) in enumerate(db):
        print(step, x_spt.shape)
    return


if __name__ == '__main__':
    #a=strides = [stride] + [1]*(num_blocks-1)
    a= [1]+[1]*(9-1)
    a= a[3::]
    print(len(a))
