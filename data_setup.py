import load_and_process as ld
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import torchvision
from torch.utils.data import DataLoader, Dataset
# Load and preprocess the data



class EmotionDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label




def create_dataloaders(batch_size,train_transform,val_transform):

    faces, emotions = ld.load_fer2013()
    faces = ld.preprocess_input(faces)
    num_samples, num_classes = emotions.shape

    xtrain, xtest, ytrain, ytest = train_test_split(faces, emotions, test_size=0.15, shuffle=True)


    train_data = EmotionDataset(xtrain, ytrain, transform=train_transform)

    test_data = EmotionDataset(xtest, ytest, transform=val_transform)

    train_loader = DataLoader(train_data, 
                            batch_size=batch_size, 
                            shuffle=True)
    test_loader = DataLoader(test_data, 
                            batch_size=batch_size, 
                            shuffle=False)
    return train_loader,test_loader
