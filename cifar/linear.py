import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from copy import deepcopy

class SimpleCoTTA(nn.Module):

    def __init__(self, model, optimizer, alpha_ema=0.9, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic

        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())

        self.ema_model = deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.requires_grad = False

        self.alpha_ema = alpha_ema

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            self.forward_and_adapt(x)

        return self.model(x)

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        preds = self.model(x)

        with torch.no_grad():
            teacher_preds = self.ema_model(x)

        loss = F.mse_loss(preds, teacher_preds)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for ema_p, student_p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.data = self.alpha_ema * ema_p.data + (1.0 - self.alpha_ema) * student_p.data

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

        self.ema_model = deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.requires_grad = False


class BinaryLinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28*28, 2)

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        return self.linear(x)


def filter_binary_class(dataset, keep_classes=(0, 1)):
    keep_idx = [i for i, (img, label) in enumerate(dataset) if label in keep_classes]
    dataset.data = dataset.data[keep_idx]
    dataset.targets = [keep_classes.index(t) for t in dataset.targets if t in keep_classes]
    return dataset


def train_one_epoch(model, dataloader, optimizer, device='cuda'):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def evaluate(model, dataloader, device='cuda'):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root='./linear_data', train=True, download=True, transform=transform)
    mnist_test  = datasets.MNIST(root='./linear_data', train=False, download=True, transform=transform)

    mnist_train = filter_binary_class(mnist_train, keep_classes=(0, 1))
    mnist_test  = filter_binary_class(mnist_test, keep_classes=(0, 1))

    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
    test_loader_source = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False)

    fmnist_test = datasets.FashionMNIST(root='./linear_data', train=False, download=True, transform=transform)
    fmnist_test = filter_binary_class(fmnist_test, keep_classes=(0, 1))

    test_loader_target = torch.utils.data.DataLoader(fmnist_test, batch_size=64, shuffle=False)

    model = BinaryLinearNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print("==> Training on MNIST (digits 0,1) ...")
    for epoch in range(1, 4):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")

    source_loss, source_acc = evaluate(model, test_loader_source, device)
    print(f"\nSource-domain evaluation (MNIST 0/1): loss={source_loss:.4f}, acc={source_acc:.4f}")

    target_loss, target_acc = evaluate(model, test_loader_target, device)
    print(f"Target-domain evaluation (FashionMNIST 0/1) without CoTTA: "
          f"loss={target_loss:.4f}, acc={target_acc:.4f}")

    print("\n==> Adapting with simplified CoTTA on target data ...")

    cotta_model = SimpleCoTTA(model, optimizer, alpha_ema=0.9, steps=1, episodic=False).to(device)

    cotta_model.model.train()
    for x, y in test_loader_target:
        x = x.to(device)
        _ = cotta_model(x)

    adapted_loss, adapted_acc = evaluate(cotta_model.model, test_loader_target, device)
    print(f"Target-domain evaluation (FashionMNIST) with CoTTA adaptation: "
          f"loss={adapted_loss:.4f}, acc={adapted_acc:.4f}")


if __name__ == '__main__':
    main()