import torch
import torch.nn as nn
import math
from typing import Optional
from model_utils import MultiHeadAttention


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_hidden: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = decoder_hidden.size(0)
        
        # Queries from decoder, keys and values from encoder
        q = self.query(decoder_hidden).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(encoder_hidden).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(encoder_hidden).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if encoder_mask is not None:
            scores = scores.masked_fill(encoder_mask == 0, float('-inf'))
            
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.proj(out)


class EncoderDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Self-attention for decoder
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention between decoder and encoder
        self.cross_attn = CrossAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_hidden: torch.Tensor,
        decoder_mask: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention on decoder
        self_attn_out = self.self_attn(decoder_hidden, decoder_mask)
        decoder_hidden = self.norm1(decoder_hidden + self.dropout(self_attn_out))
        
        # Cross-attention with encoder
        cross_attn_out = self.cross_attn(decoder_hidden, encoder_hidden, encoder_mask)
        decoder_hidden = self.norm2(decoder_hidden + self.dropout(cross_attn_out))
        
        # Feed-forward
        ff_out = self.ff(decoder_hidden)
        decoder_hidden = self.norm3(decoder_hidden + self.dropout(ff_out))
        
        return decoder_hidden


class EncoderDecoderModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_seq_length: int = 1024,
        dropout: float = 0.1,
        device: Optional[str] = None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Device selection
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
            print(f"No device specified, using best available: {self.device}")
        else:
            self.device = torch.device(device)
            print(f"Using specified device: {self.device}")
        
        # Shared token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_position_embedding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        self.decoder_position_embedding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        self.dropout = nn.Dropout(dropout)
        
        # Encoder layers (self-attention only)
        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': MultiHeadAttention(d_model, num_heads, dropout),
                'norm1': nn.LayerNorm(d_model),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model)
                ),
                'norm2': nn.LayerNorm(d_model),
                'dropout': nn.Dropout(dropout)
            })
            for _ in range(num_layers)
        ])
        
        # Decoder layers (self-attention + cross-attention + ff)
        self.decoder_layers = nn.ModuleList([
            EncoderDecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Move model to device
        self.to(self.device)
        
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        seq_len = input_ids.size(1)
        
        # Token + Position embeddings
        x = self.token_embedding(input_ids)
        x = x + self.encoder_position_embedding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Create attention mask for encoder
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_mask = extended_mask.to(dtype=torch.float)
        extended_mask = (1.0 - extended_mask) * -10000.0
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            # Self-attention
            attn_out = layer['self_attn'](x, extended_mask)
            x = layer['norm1'](x + layer['dropout'](attn_out))
            
            # Feed-forward
            ff_out = layer['ff'](x)
            x = layer['norm2'](x + layer['dropout'](ff_out))
        
        return self.encoder_norm(x)
    
    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden: torch.Tensor,
        decoder_mask: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        seq_len = decoder_input_ids.size(1)
        
        # Token + Position embeddings
        x = self.token_embedding(decoder_input_ids)
        x = x + self.decoder_position_embedding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Create causal mask for decoder if not provided
        if decoder_mask is None:
            decoder_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            decoder_mask = decoder_mask.unsqueeze(0).unsqueeze(0).to(x.device)
            decoder_mask = decoder_mask.to(dtype=torch.float)
            decoder_mask = (decoder_mask) * -10000.0
        
        # Prepare encoder mask
        if encoder_mask is not None:
            encoder_mask = encoder_mask.unsqueeze(1).unsqueeze(2)
            encoder_mask = encoder_mask.to(dtype=torch.float)
            encoder_mask = (1.0 - encoder_mask) * -10000.0
        
        # Apply decoder layers
        for layer in self.decoder_layers:
            x = layer(x, encoder_hidden, decoder_mask, encoder_mask)
        
        x = self.decoder_norm(x)
        return self.output_proj(x)
    
    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Encode input sequence
        encoder_hidden = self.encode(encoder_input_ids, encoder_attention_mask)
        
        # Decode with encoder context
        outputs = self.decode(
            decoder_input_ids, 
            encoder_hidden, 
            decoder_attention_mask,
            encoder_attention_mask
        )
        
        return outputs
    
    def generate(
        self,
        encoder_input_ids: torch.Tensor,
        decoder_start_token_id: int = 1,  # <BOS>
        max_length: int = 100,
        encoder_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self.eval()
        batch_size = encoder_input_ids.size(0)
        device = encoder_input_ids.device
        
        # Encode input sequence once
        encoder_hidden = self.encode(encoder_input_ids, encoder_attention_mask)
        
        # Initialize decoder input with start token
        generated = torch.full((batch_size, 1), decoder_start_token_id, 
                              dtype=torch.long, device=device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Decode current sequence
                outputs = self.decode(generated, encoder_hidden, 
                                    encoder_mask=encoder_attention_mask)
                
                # Get next token logits
                next_token_logits = outputs[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=1)
                
                # Stop if all sequences generated EOS token (optional)
                # if (next_token == eos_token_id).all():
                #     break
                
        return generated


# Example usage
if __name__ == "__main__":
    vocab_size = 50000
    model = EncoderDecoderModel(vocab_size=vocab_size, d_model=512, num_layers=6)
    
    # Example input
    batch_size = 2
    encoder_seq_len = 10
    decoder_seq_len = 8
    
    encoder_input = torch.randint(0, vocab_size, (batch_size, encoder_seq_len))
    decoder_input = torch.randint(0, vocab_size, (batch_size, decoder_seq_len))
    encoder_mask = torch.ones_like(encoder_input)
    
    # Forward pass
    outputs = model(encoder_input, decoder_input, encoder_mask)
    print(f"Input shapes - Encoder: {encoder_input.shape}, Decoder: {decoder_input.shape}")
    print(f"Output shape: {outputs.shape}")
    
    # Generation example
    generated = model.generate(encoder_input, max_length=20)
    print(f"Generated sequence shape: {generated.shape}")