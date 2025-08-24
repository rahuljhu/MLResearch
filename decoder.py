import torch
import torch.nn as nn
import math
from model_utils import MultiHeadAttention


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class LLMDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, 1024, d_model))
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        
        x = self.token_embedding(x)
        x = x + self.position_embedding[:, :seq_len, :]
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.norm(x)
        return self.output(x)

    def generate(self, input_ids, max_length=100):
        self.eval()
        batch_size = input_ids.size(0)
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_length):
                seq_len = generated.size(1)
                mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
                mask = mask.unsqueeze(0).unsqueeze(0)
                
                outputs = self(generated, mask=mask)
                next_token_logits = outputs[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=1)
                
        return generated

# Example usage
if __name__ == "__main__":
    vocab_size = 50000
    model = LLMDecoder(vocab_size=vocab_size)
    
    # Example input
    batch_size, seq_length = 2, 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Forward pass
    outputs = model(input_ids)
    print(f"Output shape: {outputs.shape}")
    
    # Generation example
    input_prompt = torch.tensor([[1, 2, 3]])
    generated = model.generate(input_prompt, max_length=20)
    print(f"Generated sequence shape: {generated.shape}")