from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataloading.dataloader import BasicDataset

class MakeDataset:
    def __init__(self, dataset_path, img_scale, batch_size, val_percent, msk_scale):
        base_dir = Path(dataset_path)
        dataset_name = (dataset_path.split('/'))[-1]
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None 
        self.test_dataset = None 
        self.val_percent = val_percent
        if dataset_name == 'ISIC':
            self.train_dataset = BasicDataset(images_dir=base_dir/'ISBI2016_ISIC_Part1_Training_Data', mask_dir=base_dir/'ISBI2016_ISIC_Part1_Training_GroundTruth', img_scale=img_scale, img_suffix='', mask_suffix='_Segmentation', id_dir='img', norm_mode='rgb')
            self.test_dataset = BasicDataset(images_dir=base_dir/'ISBI2016_ISIC_Part1_Test_Data', mask_dir=base_dir/'ISBI2016_ISIC_Part1_Test_GroundTruth', img_scale=img_scale, img_suffix='', mask_suffix='_Segmentation', id_dir='img', norm_mode='rgb')
        elif dataset_name == 'ISIC18':
            self.train_dataset = BasicDataset(images_dir=base_dir/'ISIC2018_Task1-2_Training_Input', mask_dir=base_dir/'ISIC2018_Task1_Training_GroundTruth', img_scale=img_scale, img_suffix='', mask_suffix='_segmentation', id_dir='img', norm_mode='rgb')
            self.test_dataset = BasicDataset(images_dir=base_dir/'ISIC2018_Task1-2_Test_Input', mask_dir=base_dir/'ISIC2018_Task1_Test_GroundTruth', img_scale=img_scale, img_suffix='', mask_suffix='_segmentation', id_dir='img', norm_mode='rgb')
        elif dataset_name == 'REFUGE2' or dataset_name == 'REFUGE2_mod':
            transform = A.Compose([
                        A.HorizontalFlip(p=0.5, fill_mask=2),
                        A.VerticalFlip(p=0.5, fill_mask=2),
                        A.Rotate(limit=45, p=0.5, fill_mask=2),
                        ToTensorV2()
                    ])
            self.train_dataset = BasicDataset(images_dir=base_dir/'train/images', mask_dir=base_dir/'train/mask', img_scale=img_scale, img_suffix='', mask_suffix='', id_dir='img', norm_mode='rgb', transform=transform)
            self.val_dataset = BasicDataset(images_dir=base_dir/'val/images', mask_dir=base_dir/'val/mask', img_scale=img_scale, img_suffix='', mask_suffix='', id_dir='img', norm_mode='rgb', transform=transform)
            self.test_dataset = BasicDataset(images_dir=base_dir/'test/images', mask_dir=base_dir/'test/mask', img_scale=img_scale, img_suffix='', mask_suffix='', id_dir='img', norm_mode='rgb', transform=None)
        elif dataset_name == 'MLUA':
            self.train_dataset = BasicDataset(images_dir=base_dir/'imagesTr', mask_dir=base_dir/'labelsTr', img_scale=img_scale, img_suffix='', mask_suffix='', id_dir='img', norm_mode='zscore')
            self.test_dataset = BasicDataset(images_dir=base_dir/'imagesTs', mask_dir=base_dir/'labelsTs', img_scale=img_scale, img_suffix='', mask_suffix='', id_dir='img', norm_mode='zscore')
        elif dataset_name == 'OCTA':
            self.train_dataset = BasicDataset(images_dir=base_dir/'imagesTr', mask_dir=base_dir/'labelsTr', img_scale=img_scale, img_suffix='', mask_suffix='', id_dir='img', norm_mode='zscore', msk_scale=msk_scale)
            self.test_dataset = BasicDataset(images_dir=base_dir/'imagesTs', mask_dir=base_dir/'labelsTs', img_scale=img_scale, img_suffix='', mask_suffix='', id_dir='img', norm_mode='zscore', msk_scale=msk_scale)
        elif dataset_name == 'CVC-ColonDB':
            self.train_dataset = BasicDataset(images_dir=base_dir/'images', mask_dir=base_dir/'masks', img_scale=img_scale, img_suffix='', mask_suffix='', id_dir='img', norm_mode='rgb')
        elif 'Dataset' in dataset_name:
            self.train_dataset = BasicDataset(images_dir=base_dir/'imagesTr', mask_dir=base_dir/'labelsTr', img_scale=img_scale, img_suffix='_0000', mask_suffix='', id_dir='mask', norm_mode='zscore') 
            self.test_dataset = BasicDataset(images_dir=base_dir/'imagesTs', mask_dir=base_dir/'labelsTs', img_scale=img_scale, img_suffix='_0000', mask_suffix='', id_dir='mask', norm_mode='zscore')
        elif dataset_name == 'PROSTATE':
            self.train_dataset = BasicDataset(images_dir=base_dir/'imagesTr', mask_dir=base_dir/'labelsTr', img_scale=img_scale, img_suffix='', mask_suffix='_segmentation', id_dir='img', norm_mode='rgb')
            self.test_dataset = BasicDataset(images_dir=base_dir/'imagesTs', mask_dir=base_dir/'labelsTs', img_scale=img_scale, img_suffix='', mask_suffix='_segmentation', id_dir='img', norm_mode='rgb')
        elif dataset_name == 'PANXRAYS':
            self.train_dataset = BasicDataset(images_dir=base_dir/'imagesTr', mask_dir=base_dir/'labelsTr', img_scale=img_scale, img_suffix='', mask_suffix='', id_dir='img', norm_mode='zscore', thresh_mask=True)
            self.test_dataset = BasicDataset(images_dir=base_dir/'imagesTs', mask_dir=base_dir/'labelsTs', img_scale=img_scale, img_suffix='', mask_suffix='', id_dir='img', norm_mode='zscore', thresh_mask=True)
        elif dataset_name == 'KVASIR':
            self.train_dataset = BasicDataset(images_dir=base_dir/'imagesTr', mask_dir=base_dir/'labelsTr', img_scale=img_scale, img_suffix='', mask_suffix='', id_dir='img', norm_mode='rgb', thresh_mask=True)
            self.test_dataset = BasicDataset(images_dir=base_dir/'imagesTs', mask_dir=base_dir/'labelsTs', img_scale=img_scale, img_suffix='', mask_suffix='', id_dir='img', norm_mode='rgb', thresh_mask=True)
        elif dataset_name == 'DRIVE':
            self.train_dataset = BasicDataset(images_dir=base_dir/'training/images', mask_dir=base_dir/'training/1st_manual', img_scale=img_scale, img_suffix='_training', mask_suffix='_manual1', id_dir='none', norm_mode='rgb', thresh_mask=True)
        elif dataset_name == 'BUSI':
            self.train_dataset = BasicDataset(images_dir=base_dir/'images', mask_dir=base_dir/'masks', img_scale=img_scale, img_suffix='', mask_suffix='_mask', id_dir='img', norm_mode='rgb')
        elif dataset_name == 'SANET':
            self.train_dataset = BasicDataset(images_dir=base_dir/'TrainDataset/image', mask_dir=base_dir/'TrainDataset/mask', img_scale=img_scale, img_suffix='', mask_suffix='', id_dir='img', norm_mode='rgb', mask_as_grayscale=True) 
            self.test_dataset = BasicDataset(images_dir=base_dir/'TestMerged/image', mask_dir=base_dir/'TestMerged/mask', img_scale=img_scale, img_suffix='', mask_suffix='', id_dir='img', norm_mode='rgb', mask_as_grayscale=True) 
        elif dataset_name == 'TESTBW':
            self.test_dataset = BasicDataset(images_dir=base_dir/'proximal_sides_images', mask_dir=base_dir/'annotations', img_scale=img_scale, img_suffix='_0000', mask_suffix='', id_dir='mask', norm_mode='zscore', mask_as_grayscale=True) 
        else:
            raise NotImplementedError('The dataset that you are working with doesn\'t have a constructor, please write one yourself!')
        self.mask_values = self.train_dataset.mask_values if self.train_dataset is not None else self.test_dataset.mask_values


    def get_loaders(self):
        if self.dataset_name == 'ISIC':
            return self.generate_without_sampler(self.val_percent)
        elif self.dataset_name == 'ISIC18':
            return self.generate_without_sampler(self.val_percent)
            # return self.generate_test()
        elif self.dataset_name == 'REFUGE2' or self.dataset_name == 'REFUGE2_mod':
            return self.generate_loaders()
        elif self.dataset_name == 'MLUA':
            return self.generate_without_sampler(self.val_percent)
        elif self.dataset_name == 'OCTA':
            return self.generate_without_sampler(self.val_percent)
        elif self.dataset_name == 'KVASIR':
            return self.generate_without_sampler(self.val_percent)
        elif self.dataset_name == 'CVC-ColonDB':
            return self.generate_val_test(self.val_percent/2)
        elif 'Dataset' in self.dataset_name:
            # return self.generate_with_sampler(self.val_percent, 1, 1)
            return self.generate_without_sampler(self.val_percent)
        elif self.dataset_name == 'PANXRAYS':
            return self.generate_without_sampler(self.val_percent)
        elif self.dataset_name == 'PROSTATE':
            return self.generate_without_sampler(self.val_percent)
            # return self.generate_with_sampler(self.val_percent, 1, 2)
        elif self.dataset_name == 'DRIVE':
            return self.generate_val_test(self.val_percent/2)
        elif self.dataset_name == 'BUSI':
            return self.generate_val_test(self.val_percent)
        elif self.dataset_name == 'SANET':
            return self.generate_without_sampler(self.val_percent)
        elif self.dataset_name == 'TESTBW':
            return self.generate_test()
        else:
            raise NotImplementedError('The dataset that you are working with doesn\'t have a loader generator, please write one yourself!')

    def generate_loaders(self):
        loader_args = dict(batch_size=self.batch_size, num_workers=0, pin_memory=True)
        train_loader = DataLoader(self.train_dataset, shuffle=True, **loader_args)
        val_loader = DataLoader(self.val_dataset, shuffle=False, drop_last=True, **loader_args)
        test_loader = DataLoader(self.test_dataset, shuffle=False, **loader_args)

        return train_loader, val_loader, test_loader
    
    def generate_with_sampler(self, val_percent, freq_e, freq_ne):
        n_val = int(len(self.train_dataset) * val_percent)
        n_train = len(self.train_dataset) - n_val
        train_set, val_set = random_split(self.train_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

        train_sampler_weights = self.train_dataset.generate_sample_freq(num_empty=freq_e, num_non_empty=freq_ne, working_set=train_set)
        train_sampler = WeightedRandomSampler(
                                weights=train_sampler_weights,
                                num_samples=len(train_sampler_weights),
                                replacement=True)
        
        
        val_sampler_weights = self.train_dataset.generate_sample_freq(num_empty=freq_e, num_non_empty=freq_ne, working_set=val_set)
        val_sampler = WeightedRandomSampler(
                                weights=val_sampler_weights,
                                num_samples=len(val_sampler_weights),
                                replacement=True)

        loader_args = dict(batch_size=self.batch_size, num_workers=0, pin_memory=True)
        train_loader = DataLoader(train_set, sampler=train_sampler, **loader_args)
        val_loader = DataLoader(val_set, sampler=val_sampler, drop_last=True, **loader_args)
        test_loader = DataLoader(self.test_dataset, shuffle=False, **loader_args)

        return train_loader, val_loader, test_loader
    
    def generate_without_sampler(self, val_percent):
        n_val = int(len(self.train_dataset) * val_percent)
        n_train = len(self.train_dataset) - n_val
        train_set, val_set = random_split(self.train_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
        loader_args = dict(batch_size=self.batch_size, num_workers=0, pin_memory=True)
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
        test_loader = DataLoader(self.test_dataset, shuffle=False, drop_last=True, **loader_args)

        return train_loader, val_loader, test_loader
    
    def generate_val_test(self, val_percent):
        n_val = int(len(self.train_dataset) * val_percent) * 2
        n_train = len(self.train_dataset) - n_val
        train_set, val_set, test_set = random_split(self.train_dataset, [n_train, n_val//2, n_val//2], generator=torch.Generator().manual_seed(0))
        loader_args = dict(batch_size=self.batch_size, num_workers=0, pin_memory=True)
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
        test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)

        return train_loader, val_loader, test_loader
    
    def generate_test(self):
        loader_args = dict(batch_size=self.batch_size, num_workers=0, pin_memory=True)
        test_loader = DataLoader(self.test_dataset, shuffle=False, drop_last=True, **loader_args)

        return test_loader