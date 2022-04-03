import os,collections
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMG_SIZE = 640

class segDataset(Dataset):
    def __init__(self, data_root, x_trans = None, y_trans = None, mode = 'train', labelme = True):
        self.root = data_root
        self.x_trans = x_trans
        self.y_trans = y_trans
        self.mode = mode
        self.files = collections.defaultdict(list)
        if x_trans is None and y_trans is None:
            self.x_trans = T.Compose([
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=[0.466, 0.464, 0.409], std=[0.269, 0.24, 0.241])
            ])
            self.y_trnas = T.Compose([
                T.Resize((IMG_SIZE,IMG_SIZE)),
                T.ToTensor()
            ])
        for split_file in ['train', 'val']:
            imgSetFiles = os.path.join(self.root,split_file+'.txt')
            with open(imgSetFiles, 'r') as f:
                data_path = f.readline()
                if labelme:
                    img_path = os.path.join(self.root, data_path.split(' ')[0],'img.png')
                    label_path = os.path.join(self.root, data_path.split(' ')[1],'label.png')
                else:
                    img_path = os.path.join(self.root, data_path.split(' ')[0])
                    label_path = os.path.join(self.root, data_path.split(' ')[1])
                self.files[split_file].append({
                    'img': img_path,
                    'label':label_path
                })
    def __getitem__(self, index):
        data = self.files[self.mode][index]
        img_path, label_path  = data['img'], data['label']
        img, label = Image.open(img_path), Image.open(label_path)
        return self.x_trans(img),self.y_trans(label)
    
    def __len__(self):
        return len(self.files[self.mode])