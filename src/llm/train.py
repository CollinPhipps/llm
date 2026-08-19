import itertools
import json
import torch
from torch.utils.data import Dataset, DataLoader
import random

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

def get_loaders(encoded_corpus_tensor, block_size, batch_size, shuffle=True):
    n = int(0.8 * len(encoded_corpus_tensor))
    train_data = encoded_corpus_tensor[:n]
    test_data = encoded_corpus_tensor[n:]

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

def train(model, dataloaders, optimizer, iters, device, save_dir, report_every=None):
    model = model.to(device)
    results = []  # (step, train_loss, val_loss)

    steps_per_epoch = len(dataloaders['train'])
    total_steps = iters * steps_per_epoch
    report_interval = total_steps // report_every if report_every else None

    step = 0
    for iter in range(iters):
        for X, Y in dataloaders['train']:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(X, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            step += 1

            if report_interval is not None and step % report_interval == 0:
                losses = estimate_loss(model, dataloaders, device)
                print(f"Step: {step}/{total_steps}, Train Loss: {losses['train']}, Val Loss: {losses['val']}")
                results.append((step, losses['train'], losses['val']))
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step}, f'{save_dir}/ckpt_step{step}.pt')
                with open(f'{save_dir}/loss_history.json', 'w') as f:
                    json.dump(results, f, indent=2)

    return results

def make_sft_examples(n_examples=200, min_len=300, min_frac=0.15, max_frac=0.30, seed=0):
    with open('corpus.txt', "r") as f:
        text = f.read()

    random.seed(seed)
    abstracts = [a for a in text.split("\n\n") if len(a) >= min_len]
    sampled = random.sample(abstracts, n_examples)

    examples = []
    for abstract in sampled:
        words = abstract.split()
        frac = random.uniform(min_frac, max_frac)
        cut = max(1, min(int(frac * len(words)), len(words) - 1))
        prompt_text = " ".join(words[:cut])
        response_text = " ".join(words[cut:])
        examples.append((prompt_text, response_text))

    return examples

class SFTDataset(Dataset):
    def __init__(self, tokenizer, examples, block_size):
        self.examples = []
        for prompt_text, response_text in examples:
            prefix = f"<|prompt|>{prompt_text}<|response|>"
            full_text = prefix + response_text

            prompt_len = len(tokenizer.encode(prefix))
            full_ids = tokenizer.encode(full_text)

            if len(full_ids) > block_size + 1:
                continue

            input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
            targets = torch.tensor(full_ids[1:], dtype=torch.long)
            targets[: prompt_len - 1] = -100

            self.examples.append((input_ids, targets))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

def get_sft_loaders(tokenizer, block_size, n_examples):
    examples = make_sft_examples(n_examples=n_examples)
    n = int(0.8 * len(examples))
    train_examples = examples[:n]
    val_examples = examples[n:]

    train_dataset = SFTDataset(tokenizer, train_examples, block_size)
    val_dataset = SFTDataset(tokenizer, val_examples, block_size)

    return {
        'train' : DataLoader(train_dataset, batch_size=1),
        'val' : DataLoader(val_dataset, batch_size=1)
    }

def sft(model, dataloaders, optimizer, epochs, device, save_dir, report_every=None):
    model = model.to(device)
    results = []  # (epoch, train_loss, val_loss)

    for epoch in range(epochs):
        for X, Y in dataloaders['train']:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(X, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if epoch == (epochs - 1):
            losses = estimate_loss(model, dataloaders, device)
            print(f"Epoch: {epoch}/{epochs}, Train Loss: {losses['train']}, Val Loss: {losses['val']}")
            results.append((epoch, losses['train'], losses['val']))
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, f'{save_dir}/final_sft.pt')
            with open(f'{save_dir}/loss_history.json', 'w') as f:
                json.dump(results, f, indent=2)

        elif report_every is not None and (epoch + 1) % report_every == 0:
            losses = estimate_loss(model, dataloaders, device)
            print(f"Epoch: {epoch}/{epochs}, Train Loss: {losses['train']}, Val Loss: {losses['val']}")
            results.append((epoch, losses['train'], losses['val']))
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, f'{save_dir}/ckpt_epoch{epoch}.pt')
            with open(f'{save_dir}/loss_history.json', 'w') as f:
                json.dump(results, f, indent=2)

    return results