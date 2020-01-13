from torch.utils.data import Dataset
from torchvision.transforms import transforms


class CIFARDATA(Dataset):
    def __init__(self, mode, n_way, k_shot, k_query, startidx=0):
        """
        :param mode: train, val or test
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param startidx: start to index label from startidx
        """
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.setsz = self.n_way*self.k_shot
        self.querysz = self.n_way*self.k_query
        self.startidx=startidx

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])