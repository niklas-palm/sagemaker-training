import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import boto3


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 1600)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def prepare_data(data_dir):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    trainset = torchvision.datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    return trainset, testset


def upload_to_s3(local_path, bucket, s3_key):
    s3 = boto3.client("s3")
    try:
        s3.upload_file(local_path, bucket, s3_key)
        print(f"Successfully uploaded {local_path} to s3://{bucket}/{s3_key}")
    except Exception as e:
        print(f"Failed to upload to S3: {str(e)}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    trainset, testset = prepare_data(args.data_dir)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False
    )

    # Model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        for data in trainloader:
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}, Accuracy: {100 * correct / total}%")

    # Save model locally
    os.makedirs(args.model_dir, exist_ok=True)
    local_model_path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.state_dict(), local_model_path)

    # Upload model to S3
    if args.s3_bucket and args.s3_key:
        upload_to_s3(local_model_path, args.s3_bucket, args.s3_key)


if __name__ == "__main__":

    data_dir = "./data"
    model_dir = "./model"
    s3_bucket = "mlbucket13"
    s3_key = "mymodel/model.pt"
    epochs = 2
    batch_size = 64

    args = argparse.Namespace(
        epochs=epochs,
        model_dir=model_dir,
        data_dir=data_dir,
        s3_bucket=s3_bucket,
        s3_key=s3_key,
        batch_size=batch_size,
    )

    train(args)
