from torchvision.utils import save_image
import os
from torchvision import datasets, transforms
import torch
import numpy as np
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CIFAR10Pipeline(Pipeline):
    def __init__(self, data_dir, batch_size, num_threads, device_id, training=True):
        super().__init__(batch_size, num_threads, device_id, seed=42)
        self.data_dir = data_dir
        self.training = training

    def define_graph(self):
        images, labels = fn.readers.file(
            file_root=self.data_dir,
            random_shuffle=self.training,
            name="Reader"
        )
        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
        images = fn.resize(images, resize_x=224, resize_y=224)
        images = fn.crop_mirror_normalize(
            images,
            device="gpu",
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[0.5071 * 255, 0.4867 * 255, 0.4408 * 255],
            std=[0.2675 * 255, 0.2565 * 255, 0.2761 * 255]
        )
        return images, labels


class Nvidia_DALI_Dataloader():
    def __init__(self, root_dir, batch_size, train_ratio=0.9, shuffle=True, seed=42):
        self.root_dir = root_dir
        self.train_ratio = train_ratio
        self.batch_size = batch_size

        os.makedirs(os.path.join(self.root_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, 'valid'), exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, 'test'), exist_ok=True)

        transform = transforms.ToTensor()
        train_dataset = datasets.CIFAR10(
            root=root_dir,
            train=True,
            download=True,
            transform=transform)
        test_dataset = datasets.CIFAR10(
            root=root_dir,
            train=False,
            download=True,
            transform=transform)
        class_names = train_dataset.classes

        # Save test set
        for idx, (img, label) in enumerate(test_dataset):
            class_dir = os.path.join(self.root_dir, 'test', class_names[label])
            os.makedirs(class_dir, exist_ok=True)
            save_image(img, os.path.join(class_dir, f'{idx}.png'))
        print(f"Saved {len(test_dataset)} images to {os.path.join(self.root_dir, 'test')}")

        # Split and save train/valid
        np.random.seed(seed)
        indices = np.random.permutation(len(train_dataset))
        split_idx = int(train_ratio * len(train_dataset))
        train_indices = indices[:split_idx]
        valid_indices = indices[split_idx:]

        for idx in train_indices:
            img, label = train_dataset[idx]
            class_dir = os.path.join(root_dir, 'train', class_names[label])
            os.makedirs(class_dir, exist_ok=True)
            save_image(img, os.path.join(class_dir, f'{idx}.png'))

        for idx in valid_indices:
            img, label = train_dataset[idx]
            class_dir = os.path.join(root_dir, 'valid', class_names[label])
            os.makedirs(class_dir, exist_ok=True)
            save_image(img, os.path.join(class_dir, f'{idx}.png'))

        print(f"Saved {len(train_indices)} images to 'train/' and {len(valid_indices)} to 'valid/'.")

        self.train_directory = os.path.join(self.root_dir, 'train')
        self.valid_directory = os.path.join(self.root_dir, 'valid')
        self.test_directory = os.path.join(self.root_dir, 'test')

    def get_dali_loader(self):
        def make_loader(data_dir, training):
            pipeline = CIFAR10Pipeline(
                data_dir=data_dir,
                batch_size=self.batch_size,
                num_threads=4,
                device_id=0,
                training=training
            )
            pipeline.build()
            loader = DALIGenericIterator(
                pipeline,
                output_map=["data", "label"],
                size=pipeline.epoch_size("Reader"),
                auto_reset=True
            )
            return loader

        train_loader = make_loader(self.train_directory, training=True)
        valid_loader = make_loader(self.valid_directory, training=False)
        test_loader = make_loader(self.test_directory, training=False)

        return train_loader, valid_loader, test_loader
