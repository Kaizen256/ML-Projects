import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def main():
    transform = transforms.ToTensor()
    dataset = datasets.ImageFolder(root='tiny-imagenet-200/train', transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images, _ in loader:
        images = images.float()  # ensure float
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    print('Mean:', mean)
    print('Std:', std)

if __name__ == "__main__":
    main()
