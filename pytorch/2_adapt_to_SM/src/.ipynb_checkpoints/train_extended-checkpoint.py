import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data from input channels
    train_data_path = os.path.join(args.train_dir, 'train_dataset.pt')
    test_data_path = os.path.join(args.test_dir, 'test_dataset.pt')
    
    trainset = torch.load(train_data_path)
    testset = torch.load(test_data_path)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    
    # Model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    for epoch in range(args.epochs):
        print(f'--- Epoch {epoch} ---')
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:  # Log every 100 mini-batches
                print(f"train_loss: {running_loss/100:.4f}")
                running_loss = 0.0
        
        # Validation
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

        accuracy = 100 * correct / total
        avg_test_loss = test_loss / len(testloader)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f'--- Epoch {epoch} test results: ---')
        print(f"test_accuracy: {accuracy:.4f}")
        print(f"test_loss: {avg_test_loss:.4f}")
        print(f"f1_score: {f1:.4f}")
    
    # Save model
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))

    # Create summary.txt
    summary_path = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Model: SimpleCNN\n")
        f.write(f"Epochs trained: {args.epochs}\n")
        f.write(f"Final test accuracy: {accuracy:.4f}\n")
        f.write(f"Final test loss: {avg_test_loss:.4f}\n")
        f.write(f"Final F1 score: {f1:.4f}\n")
    
    print(f"Summary file created at: {summary_path}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)

    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--test-dir', type=str, default=os.environ['SM_CHANNEL_TESTING'])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    
    args = parser.parse_args()
    train(args)