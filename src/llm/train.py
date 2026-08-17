import itertools
import torch
from torch.utils.data import Dataset, DataLoader

class TokenDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y

def get_loaders(encoded_corpus, block_size, batch_size, shuffle=True):
    data = torch.tensor(encoded_corpus, dtype=torch.long)
    n = int(0.8 * len(data))
    train_data = data[:n]
    test_data = data[n:]

    train_dataset = TokenDataset(train_data, block_size)
    val_dataset = TokenDataset(test_data, block_size)

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle),
    }

@torch.no_grad()
def estimate_loss(model, dataloaders, device, eval_iters=200):
    out = {}
    model.eval()

    for split, loader in dataloaders.items():
        total_loss = 0
        total_samples = 0

        for X, Y in itertools.islice(loader, eval_iters):
            X, Y = X.to(device), Y.to(device)
            _, loss = model(X, Y)

            batch_size = X.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        out[split] = total_loss / total_samples

    model.train()
    return out

def train(model, dataloaders, optimizer, epochs, device, report_every=None):
    model = model.to(device)
    results = [] # (epoch, train_loss, val_loss)

    for epoch in range(epochs):

        if report_every is not None and (epoch + 1) % (epochs // report_every) == 0:
            losses = estimate_loss(model, dataloaders, device)
            print(f"Epoch: {epoch}, Train Loss: {losses['train']}, Val Loss: {losses['val']}")
            results.append((epoch, losses['train'], losses['val']))

        for X, Y in dataloaders['train']:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(X, Y)
            loss.backward()
            optimizer.step()

    return results