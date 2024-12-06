import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import f1_score

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

def train(args):
    # Initialize process group
    dist.init_process_group(backend='gloo')  # gloo backend typically used for CPU nccl
    #dist.init_process_group(backend='nccl')  # nccl backend typically used for GPU 
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Process rank {rank}/{world_size-1} initialized")

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    # Load data from input channels
    train_data_path = os.path.join(args.train_dir, "train_dataset.pt")
    test_data_path = os.path.join(args.test_dir, "test_dataset.pt")

    trainset = torch.load(train_data_path)
    testset = torch.load(test_data_path)

    train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, sampler=train_sampler, shuffle=False
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False
    )

    model = SimpleCNN().to(device)
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    print(f"Rank {rank}: Dataset size: {len(trainset)}")
    print(f"Rank {rank}: Batch size per GPU: {args.batch_size}")
    print(f"Rank {rank}: Number of batches: {len(trainloader)}")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_sampler.set_epoch(epoch)
        
        print(f"Rank {rank}: Starting epoch {epoch+1}")
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99 and rank == 0:
                print(f"Rank {rank}: Epoch {epoch+1}, Batch {i+1}, train_loss: {running_loss/100:.4f}")
                running_loss = 0.0

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Gather results from all processes
        total = torch.tensor(total).to(device)
        correct = torch.tensor(correct).to(device)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)

        accuracy = 100 * correct.item() / total.item()
        avg_test_loss = test_loss / len(testloader)
        f1 = f1_score(all_labels, all_preds, average="weighted")

        if rank == 0:
            print(f"--- Epoch {epoch+1} test results: ---")
            print(f"test_accuracy: {accuracy:.4f}")
            print(f"test_loss: {avg_test_loss:.4f}")
            print(f"f1_score: {f1:.4f}")

    if rank == 0:
        torch.save(model.module.state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./"))
    parser.add_argument("--train-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "./"))
    parser.add_argument("--test-dir", type=str, default=os.environ.get("SM_CHANNEL_TESTING", "./"))
    parser.add_argument("--output-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "./"))

    args = parser.parse_args()
    train(args)