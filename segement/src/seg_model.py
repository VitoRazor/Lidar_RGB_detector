import argparse
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from .models.net import EncoderDecoderNet, SPPNet
#from .dataset.cityscapes import CityscapesDataset
from .utils.preprocess import minmax_normalize

valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
id2cls_dict = dict(zip(range(19), valid_classes))
id2cls_func = np.vectorize(id2cls_dict.get)
        
# parser = argparse.ArgumentParser()
# #parser.add_argument('config_path')
# parser.add_argument('--tta', action='store_true')
# parser.add_argument('--vis', action='store_true')
# args = parser.parse_args()
#config_path = Path(args.config_path)
config_path = Path("../segement/config/eval/cityscapes_deeplab_v3_plus.yaml")
tta_flag = True
vis_flag = False
tta_flag = True
config = yaml.load(open(config_path))
net_config = config['Net']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
modelname = config_path.stem
model_path = Path('../segement/model') / modelname / 'model.pth'

# def predict(batched, tta_flag=False):
#     #images, labels, names = batched
#     images , names = batched
#     images_np = images.numpy().transpose(0, 2, 3, 1)
#     #labels_np = labels.numpy()

#     #images, labels = images.to(device), labels.to(device)
#     images = images.to(device)
#     if tta_flag:
#         preds = model.tta(images, scales=scales, net_type=net_type)
#     else:
#         preds = model.pred_resize(images, images.shape[2:], net_type=net_type)
#     preds = preds.argmax(dim=1)
#     preds_np = preds.detach().cpu().numpy().astype(np.uint8)
#     return images_np, None, preds_np, names
class seg_model():
    def __init__(self):
        if 'unet' in net_config['dec_type']:
            self.net_type = 'unet'
            self.model = EncoderDecoderNet(**net_config)
        else:
            self.net_type = 'deeplab'
            self.model = SPPNet(**net_config)
        self.model.to(device)
        self.model.update_bn_eps()

        param = torch.load(model_path)
        self.model.load_state_dict(param)
        del param
        self.model.eval()
        batch_size = 1
        self.scales = [0.25, 0.75, 1, 1.25]
        #valid_dataset = CityscapesDataset(split='test',kitti = "G:/3D_detector/data", net_type=net_type)
        #valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    def fix_img(self, img_path):
        img = np.array(Image.open(img_path))
        # Resize (Scale & Pad & Crop)
        img = minmax_normalize(img, norm_range=(-1, 1))
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis,:]
        img = torch.FloatTensor(img)
        return img
    def predict(self, img_path, vis = False):
        with torch.no_grad():
            images = self.fix_img(img_path)
            name = img_path.stem
            #print(images.shape)
            images_np = images.numpy().transpose(0, 2, 3, 1)
            #labels_np = labels.numpy()

            #images, labels = images.to(device), labels.to(device)
            images = images.to(device)
            if tta_flag:
                preds = self.model.tta(images, scales=self.scales, net_type=self.net_type)
            else:
                preds = self.model.pred_resize(images, images.shape[2:], net_type=self.net_type)
            
            out = preds.detach().cpu().numpy()
            #print(out[0,:,0,1:5])
            out = out.astype(np.float32)
            preds = preds.argmax(dim=1)
            preds_np = preds.detach().cpu().numpy().astype(np.uint8)
            
            if vis:
                preds_show = id2cls_func(preds_np).astype(np.uint8)
                #preds_np = label_to_color_image(preds_np[0]).astype(np.int32)
                fig, axes = plt.subplots(1, 2, figsize=(70,30))
                plt.tight_layout()
                axes[0].set_title('input image')
                axes[1].set_title('prediction')
                axes[0].imshow(minmax_normalize(images_np[0], norm_range=(0, 1), orig_range=(-1, 1)))
                axes[1].imshow(preds_show[0])
                plt.savefig("result/" + name + '.png')
                plt.close()
            return out[0].transpose(1, 2, 0)