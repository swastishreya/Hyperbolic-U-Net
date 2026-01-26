import logging
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import nibabel as nib
from functools import partial
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename, load_img_as_gray=False):
    ext = splitext(filename)[1]
    ext2 = splitext(splitext(filename)[0])[1] + splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    elif ext in ['.nii', '.nii.gz'] or ext2 in ['.nii', '.nii.gz']:
        return nib.load(filename).get_fdata()
    else:
        if load_img_as_gray:
            print("LOADING GRAY IMAGE")
            return Image.open(filename).convert("L")
        return Image.open(filename)

def load_as_grayscale(mask_path: Path):
    return Image.open(mask_path).convert("L")


def unique_mask_values(idx, mask_dir, mask_suffix, thresh_mask=False, mask_as_grayscale=False):
    try:
        mask_file = list(mask_dir.glob(idx + mask_suffix +'.*'))[0]
        if mask_as_grayscale:
            mask = np.asarray(load_as_grayscale(mask_file))
        else:
            mask = np.asarray(load_image(mask_file))
        if thresh_mask:
            mask = mask / 255.
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
        if mask.ndim == 2:
            return np.unique(mask)
        elif mask.ndim == 3:
            mask = mask.reshape(-1, mask.shape[-1])
            return np.unique(mask, axis=0)
        else:
            raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')
    except IndexError:
        raise RuntimeError(f'No mask found for {idx} in {mask_dir} with suffix {mask_suffix}. Make sure the mask files are named correctly and exist in the directory.') from None
    except Exception as e:
        raise RuntimeError(f'Error processing mask for {idx}: {e}') from None


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, img_scale: float = 1.0, img_suffix: str = '', mask_suffix: str = '', id_dir='mask', norm_mode='zscore', thresh_mask=False, transform=None, msk_scale=None, mask_as_grayscale=False, img_as_grayscale=False):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < img_scale <= 1, 'Scale must be between 0 and 1'
        if msk_scale is not None:
            assert 0 < msk_scale <= 1, 'Scale must be between 0 and 1'
        self.img_scale = img_scale
        self.msk_scale = msk_scale
        self.norm_mode = norm_mode
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.thresh_mask = thresh_mask
        self.mask_as_grayscale = mask_as_grayscale
        self.transform = transform
        self.img_as_grayscale = img_as_grayscale

        if id_dir == 'mask':
            self.ids = [splitext(file)[0] for file in listdir(mask_dir) if isfile(join(mask_dir, file)) and not file.startswith('.')]
        elif id_dir == 'img':
            self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        elif id_dir == 'none':
            self.ids = [(splitext(file)[0]).rstrip(img_suffix) for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        else:
            raise NotImplementedError
        if not self.ids:
            if id_dir == 'mask':
                raise RuntimeError(f'No input file found in {mask_dir}, make sure you put your images there')
            elif id_dir == 'img':
                raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
            else:
                raise NotImplementedError

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=mask_suffix, thresh_mask=self.thresh_mask, mask_as_grayscale=self.mask_as_grayscale), self.ids),
                total=len(self.ids)
            ))
        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def pad_to_square_np(image_array, value):
        current_height, current_width = image_array.shape[-2:]
        
        max_dim = max(current_height, current_width)
        
        if current_height < max_dim:
            pad_height = max_dim - current_height
            padding = ((0, pad_height), (0, 0))  
        
        elif current_width < max_dim:
            pad_width = max_dim - current_width
            padding = ((0, 0), (0, pad_width)) 
        
        else:
            return image_array 
        
        if len(image_array.shape) == 3:
            padding = ((0, 0), *padding)
        
        padded_image = np.pad(image_array, padding, mode='constant', constant_values=value)

        return padded_image
    
    def generate_sample_freq(self, num_empty, num_non_empty, working_set=None):
        sample_freq = []
        new_ids = [working_set.dataset.ids[i] for i in working_set.indices]
        for name in new_ids:
            mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
            assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
            mask = load_image(mask_file[0])
            mask, sum = self.preprocess(self.mask_values, mask, self.img_scale, is_mask=True)
            if sum == 0.0:
                sample_freq.append(num_empty)
            else:
                sample_freq.append(num_non_empty)

        return sample_freq


    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask, norm_mode='zscore', thresh_mask=False):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            if thresh_mask:
                img = img / 255.
                img[img > 0.5] = 1
                img[img <= 0.5] = 0
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            mask = BasicDataset.pad_to_square_np(mask, mask.min())
            return mask, np.mean(mask)

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                if norm_mode == 'zscore':
                    mean = np.mean(img)
                    std = np.std(img)
                    img = (img - mean)/std
                elif norm_mode == 'rgb':
                    img = img / 255.0
                img = BasicDataset.pad_to_square_np(img, img.min())

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + self.img_suffix + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        if self.norm_mode == 'rgb':
            mask = load_image(mask_file[0])
            img = load_image(img_file[0])
        else:
            mask = load_image(mask_file[0])
            img = load_image(img_file[0], self.img_as_grayscale)

        img = self.preprocess(self.mask_values, img, self.img_scale, is_mask=False, norm_mode=self.norm_mode)
        if self.msk_scale is not None:
            mask, mean = self.preprocess(self.mask_values, mask, self.msk_scale, is_mask=True, norm_mode=self.norm_mode, thresh_mask=self.thresh_mask)
        else:
            mask, mean = self.preprocess(self.mask_values, mask, self.img_scale, is_mask=True, norm_mode=self.norm_mode, thresh_mask=self.thresh_mask)
        mean_tensor = torch.as_tensor(mean)
        

        if self.transform:
            img = np.transpose(img, axes=(1, 2, 0)) 
            augmented = self.transform(image=img, mask=mask)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask']
            
            image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=(256,256), mode='bicubic').squeeze(0)
            mask_tensor = mask_tensor.squeeze(-1)
            mask_tensor = mask_tensor.unsqueeze(0).to(torch.uint8)
            mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=(256,256), mode='nearest').squeeze(0).squeeze(0)

            return {
                'image': image_tensor,
                'mask': mask_tensor.long(),
                'mask_mean': mean_tensor,
                'image_name': name
            }

        else:
            image_tensor = torch.as_tensor(img.copy()).float().contiguous()
            image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=(256,256), mode='bicubic').squeeze(0)
            mask_tensor = torch.as_tensor(mask.copy()).float().contiguous()
            mask_tensor = F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0), size=(256,256), mode='nearest').squeeze(0).squeeze(0)

            return {
                'image': image_tensor,
                'mask': mask_tensor.long(),
                'mask_mean': mean_tensor,
                'image_name': name
            }