from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
import os
from PIL import Image

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Load image paths and labels
        for label, category in enumerate(['original', 'fake']):
            for x in range(1, 11):
                category_dir = os.path.join(data_dir, category, str(x))
                for root, _, files in os.walk(category_dir):
                    for file in files:
                        if file.endswith('.png'):
                            self.image_paths.append(os.path.join(root, file))
                            self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def read_deepfake_dataset(batchsize, data_dir):

    transform_train = transforms.Compose([
                                    transforms.Resize((32, 32)),  # Resize all images to 32x32
                                    transforms.RandomCrop(32, padding=4),  
                                    transforms.RandomHorizontalFlip(p=0.5), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])])

    transform_test = transforms.Compose([
                                    transforms.Resize((32, 32)),  # Resize all images to 32x32
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])])

    # Load the full dataset
    full_dataset = DeepfakeDataset(data_dir=data_dir, transform=None)

    # Calculate split sizes
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Apply transforms to the split datasets
    train_dataset.dataset.transform = transform_train
    test_dataset.dataset.transform = transform_test

    data_loader_train = DataLoader(dataset=train_dataset,
                                   batch_size=batchsize,
                                   shuffle=True,
                                   pin_memory=True)
    data_loader_test = DataLoader(dataset=test_dataset,
                                  batch_size=batchsize,
                                  shuffle=False,
                                  pin_memory=True)
    return data_loader_train, data_loader_test
