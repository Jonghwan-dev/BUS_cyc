import os
import random
from PIL import Image
from .base_dataset import BaseDataset, get_transform
from .image_folder import make_dataset

class SweepDataset(BaseDataset):
    """This dataset class can load unaligned/unpaired datasets with train/val/test split for sweep experiments."""

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, 'trainA')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, 'trainB')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        
        # transform 초기화
        self.transform = get_transform(opt, grayscale=(opt.input_nc == 1))
        
        # 데이터 분할 비율 설정
        self.train_ratio = 0.9  # 전체 데이터의 90%를 train+val로 사용
        self.val_ratio = 0.2    # train+val 중 20%를 validation으로 사용
        
        # 전체 인덱스 생성 및 섞기
        indices = list(range(self.A_size))
        random.seed(42)  # 재현성을 위한 시드 설정
        random.shuffle(indices)
        
        # train+val과 test로 분할
        train_val_size = int(self.A_size * self.train_ratio)
        self.train_val_indices = indices[:train_val_size]
        self.test_indices = indices[train_val_size:]
        
        # train+val을 train과 val로 분할
        val_size = int(len(self.train_val_indices) * self.val_ratio)
        self.train_indices = self.train_val_indices[val_size:]
        self.val_indices = self.train_val_indices[:val_size]
        
        # 현재 phase에 따라 사용할 인덱스 선택
        if opt.phase == 'train':
            self.indices = self.train_indices
        elif opt.phase == 'val':
            self.indices = self.val_indices
        else:  # test phase
            self.indices = self.test_indices

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        B_path = self.B_paths[index % self.B_size]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform(A_img)
        B = self.transform(B_img)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.indices)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser 