from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import kagglehub
import shutil
import os


class DataSetManager:
    def __init__(self, kaggle_dataset_name="masoudnickparvar/brain-tumor-mri-dataset",
                 raw_data_dir="./raw/brain_tumor_mri_dataset/"):
        """

        :param kaggle_dataset_name: Name of the kaggle dataset to download
        :param raw_data_dir: Path of the directory to store the dataset in
        """
        self.kaggle_dataset_name = kaggle_dataset_name
        self.raw_data_dir = raw_data_dir
        self.raw_train_data = raw_data_dir + "/Training"
        self.raw_test_data = raw_data_dir + "/Testing"

    def download_and_move_dataset(self):
        """
        Downloads the kaggle dataset by calling Kaggle's API with the dataset name, and moves it to 'self.raw_data_dir'
        :return: Path where the dataset is stored
        """
        # No need to download the dataset if it's already been downloaded and exists
        if os.path.exists(self.raw_data_dir):
            print(f"Dataset already exists at: {self.raw_data_dir}. Skipping download.")
            return self.raw_data_dir

        print("Downloading dataset...")
        path = kagglehub.dataset_download(self.kaggle_dataset_name)
        print(f"Dataset downloaded to: {path}")

        # Move the dataset to 'data/raw/brain_tumor_mri_dataset/' for easy reach
        print(f"Moving dataset to: {self.raw_data_dir}")
        shutil.copytree(path, self.raw_data_dir)
        print(f"Data successfully moved to: {self.raw_data_dir}")

        return self.raw_data_dir

    def load_data(self, batch_size=16):
        """
        Load dataset
        :param batch_size: Splits datasets into mini batches defined by 'batch_size'
        :return: train_loader, val_loader, and test_loader (preprocessed datasets)
        """

        train_transform = transforms.Compose([
            # Resize all images to (224 x 224) for consistency
            transforms.Resize((224, 224)),
            # Randomly flip some of the images both horizontally and vertically
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # Randomly rotates some images between [-10,+10] degrees, also provides variations to the model
            transforms.RandomRotation(10),
            # Convert the image into a PyTorch Tensor
            transforms.ToTensor(),
            # Normalize each channel to prevent biases and improve generality of the model
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load both the training and testing datasets
        full_train_dataset = datasets.ImageFolder(root=self.raw_train_data)
        full_test_dataset = datasets.ImageFolder(root=self.raw_test_data)

        # Split into train and validation sets
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

        # Apply the transformations
        train_dataset.transform = train_transform
        val_dataset.transform = val_test_transforms
        full_test_dataset.transform = val_test_transforms

        # Create batches for both train and validation datasets
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset=full_test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
