import os 
from PIL import Image
import torch.utils
import torch.utils.data
from utils import *
import numpy as np
from torch.utils.data import Dataset
from config import (
    train_transform,
    ROOT_DIR,
    PAD_MIRRORING,
    IMAGE_HEIGHT, 
    IMAGE_WIDTH
)

class DAVIS2017(Dataset):
    def __init__(self, train = True,
                 db_root_dir = ROOT_DIR,
                 transform = None,
                 seq_name = None,
                 pad_mirroring = None):
        self.train = train
        self.db_root_dir = db_root_dir        
        self.transform = transform
        self.seq_name = seq_name
        self.pad_mirroring = pad_mirroring
        
        if self.train == 1:
            fname = 'train'
        elif self.train == 0:
            fname = 'val'
        else: 
            fname = 'test-dev'
            
        if self.seq_name is None: 
            
            with open(os.path.join(db_root_dir, "ImageSets/2017", fname + '.txt' )) as f:
                seqs = f.readlines()
                img_list = []
                labels = []
                for seq in seqs:
                    images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', seq.strip())))
                    images_path = list(map(lambda x: os.path.join('JPEGImages/480p/', seq.strip(),x), images))
                    img_list.extend(images_path)
                    
                    lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', seq.strip())))
                    lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq.strip(),x), lab))
                    labels.extend(lab_path)
                    
        else: 
            names_images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', str(seq_name))))
            
            img_list = list(map(lambda x: os.path.join(db_root_dir, 'JPEGImages/Full-Resolution/', str(seq_name), x), names_images))
            name_label = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', str(seq_name))))
            
            labels = [os.path.join('Annotations/480p/', str(seq_name), name_label[0])]
            
            if self.train:
                img_list = [img_list[0]]
                labels = [ labels[0] ]
                
        print(len(labels), len(img_list))
        assert (len(labels) == len(img_list))
        
        self.img_list = img_list
        self.labels = labels 
        
        print('Completed initialization' + fname + ' Dataset')           
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img = np.array(Image.open(os.path.join(self.db_root_dir, self.img_list[idx])).convert("RGB"), dtype=np.float32)
        gt = np.array(Image.open(os.path.join(self.db_root_dir, self.labels[idx])).convert("L"), dtype=np.float32)
        
        gt = ((gt/np.max([gt.max(), 1e-8])) > 0.5).astype(np.float32)
        
        if self.transform is not None:
            
            augmentations =  self.transform(image = img, mask = gt)
            img = augmentations['image']
            gt = augmentations['mask']
            
        if self.pad_mirroring:
            img = Pad(padding = self.pad_mirroring, padding_mode = "reflect")(img)
            
        return img, gt

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))
        return list(img.shape[:2])
    
if __name__ == '__main__': 
    import torch
    from matplotlib import pyplot as plt
    from utils import inv_normalize, tens2image
    
    dataset = DAVIS2017(db_root_dir=ROOT_DIR, train=True, transform=train_transform, pad_mirroring=PAD_MIRRORING)
    
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=1)
    assert getattr(dataloader, "batch_size") == 1
    
    for i, (img, gt) in enumerate(dataloader):
        img = CenterCrop((IMAGE_HEIGHT, IMAGE_WIDTH))(img)
        plt.figure()
        plt.imshow(overlay_mask(inv_normalize(tens2image(img)), tens2image(gt)))
        
        if i == 20:
            break
        
    plt.show(block=True)