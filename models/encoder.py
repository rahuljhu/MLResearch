import torch
import torch.nn as nn
import math
from typing import Optional
from .model_utils import MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
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
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention block
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward block
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class LLMEncoder(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_seq_length: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        seq_len = x.size(1)
        
        # Token + Position embeddings
        x = self.token_embedding(x)
        x = x + self.position_embedding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Apply encoder layers
        for layer in self.layers:
            x = layer(x, mask)
            
        return self.norm(x)
    
    def encode_sequence(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode a sequence of input tokens into contextual embeddings.
        
        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Mask for padding tokens
            
        Returns:
            torch.Tensor: Contextual embeddings [batch_size, seq_len, d_model]
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        # Create attention mask for self-attention
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_mask = extended_mask.to(dtype=torch.float)
        extended_mask = (1.0 - extended_mask) * -10000.0
        
        return self.forward(input_ids, extended_mask)

# Example usage
if __name__ == "__main__":
    # Initialize encoder
    vocab_size = 50000
    encoder = LLMEncoder(vocab_size=vocab_size)
    
    # Example input
    batch_size, seq_length = 2, 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    
    # Encode sequence
    encoded = encoder.encode_sequence(input_ids, attention_mask)
    print(f"Input shape: {input_ids.shape}")
    print(f"Encoded shape: {encoded.shape}")
    
    # Test with padding
    input_ids_padded = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask_padded = torch.tensor([
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    ])
    
    encoded_padded = encoder.encode_sequence(input_ids_padded, attention_mask_padded)
    print(f"\nPadded encoding shape: {encoded_padded.shape}")