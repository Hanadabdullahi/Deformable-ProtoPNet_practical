import os
import pandas as pd
from PIL import Image  # For working with images
from torch.utils.data import Dataset  # Importing the base Dataset class from PyTorch
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms

class Med(Dataset):
    """ Medical Images Dataset.
    Args:
        root (string): Root directory of dataset where directories 'train' and 'test' are located, or a specific image path.
        train (bool, optional): If true, creates dataset from training set, otherwise from test set.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        self.classes = ["bacterial pneumonia", "normal", "viral pneumonia"]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        if os.path.isfile(root):
            self.samples = [(root, self.get_class_index(root))]
        else:
            self.data_folder = 'train' if self.train else 'test'
            self.samples = self.load_samples()
        """
        Vad som händer här:
        root = vägen till där mappen finns
        train = om den är sann så använder vi träningsdatan, annars så använder vi den vanliga träningsdatan
        classes = våra klasser som finns
        class to indx = 
        """

    def get_class_index(self, image_path):
        parent_folder = os.path.basename(os.path.dirname(image_path))
        return self.class_to_idx.get(parent_folder, -1)

    def load_samples(self):
        samples = []
        for class_name in self.classes:
            class_index = self.class_to_idx[class_name]
            class_path = os.path.join(self.root, self.data_folder, class_name)

            if not os.path.isdir(class_path):
                continue

            for image_name in os.listdir(class_path):
                if image_name.endswith('.jpeg'):
                    image_path = os.path.join(class_path, image_name)
                    samples.append((image_path, class_index))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target_class = self.samples[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, target_class

def preprocess_image(img_path):
    """Preprocess a single image path into a tensor."""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Adjust size as needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image)
    return torch.unsqueeze(image_tensor, 0)  # Adds a batch dimension