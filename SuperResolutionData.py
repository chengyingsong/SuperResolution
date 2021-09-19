from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
import os.path as osp
from PIL import Image, ImageFilter
from torchvision.transforms import ToPILImage


class SRData(Dataset):
    def __init__(self, data_folder=r"D:\study\项目\超分\BSDS300\images", subset="train", transform=None, demo=False):
        self.img_paths = sorted(glob(osp.join(data_folder + "/" + subset, "*.jpg")))
        self.subset = subset
        self.demo = demo

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __getitem__(self, index):
        # 将高清图片转换为YCbCr
        high = (
            Image.open(self.img_paths[index]).resize([256, 256]).convert("YCbCr")
        )
        topil = ToPILImage()
        high_y, high_cb, high_cr = high.split()

        # print(high.size)
        low = high.filter(ImageFilter.BLUR())
        # print(low.size)
        low_y, low_cb, low_cr = low.split()

        totensor = transforms.ToTensor()
        low_rgb = Image.merge("YCbCr", [topil(totensor(low_y)), low_cb, low_cr]).convert("RGB")
        high_rgb = Image.merge("YCbCr", [topil(totensor(high_y)), high_cb, high_cr]).convert("RGB")

        if self.subset == "train":
            return self.transform(low_rgb), self.transform(high_rgb)
        else:
            return low_rgb, high_rgb
        # if self.subset == "train":
        #     if self.demo:
        #         return (
        #             self.transform(low_y),
        #             self.transform(high_y),
        #             (high_cb, high_cr, low_cb, low_cr)
        #         )
        #     else:
        #         return self.transform(low_y), self.transform(high_y)
        # else:
        #     totensor = transforms.ToTensor()
        #     if self.demo:
        #         return (
        #             totensor(low_y),
        #             totensor(high_y),
        #             (high_cb, high_cr, low_cb, low_cr)
        #         )
        #     else:
        #         return totensor(low_y), totensor(high_y)

    def __len__(self):
        return len(self.img_paths)
