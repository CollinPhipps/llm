import torch
import torch.nn as nn
from torch.nn import functional as F

n_layer = 6
d_model = 384
n_head = 6
d_head = 64
block_size = 384
dropout = 0.2
vocab_size = 5000

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(d_model, d_head, bias=False)
        self.query = nn.Linear(d_model, d_head, bias=False)
        self.value = nn.Linear(d_model, d_head, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(n_head)])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
    )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = MultiHeadAttention()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.self_attention(self.ln1(x))
        return x + self.ffwd(self.ln2(x))

class LanguageModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def _time_embedding(self, t):
        # t shape: (T,) -> positions e.g., torch.arange(T, device=device)
        div_term = 10_000 ** (torch.arange(0, d_model, 2, device=self.device).float() / d_model)
        
        args = t.unsqueeze(1) / div_term
        
        sin_emb = torch.sin(args)
        cos_emb = torch.cos(args)
        
        pos_emb = torch.zeros(t.size(0), d_model, device=self.device)
        pos_emb[:, 0::2] = sin_emb
        pos_emb[:, 1::2] = cos_emb
        
        return pos_emb

    def forward(self, x, targets=None):
        B, T = x.shape
        tok_emb = self.token_embedding_table(x)
        pos_emb = self._time_embedding(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, x, max_new_tokens, temp=1):
        self.eval()
        for _ in range(max_new_tokens):
            x_cond = x[:, -block_size:]
            logits, loss = self(x_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits / temp, dim=-1)
            next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next), dim=1)
        self.train()
        return x