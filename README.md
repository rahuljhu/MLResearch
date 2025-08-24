# MLResearch - Transformer Language Model Implementation

A PyTorch implementation of transformer-based language models with encoder-decoder architecture and custom tokenization.

## Components

### Encoder (`encoder.py`)
- **EncoderBlock**: Self-attention block with layer normalization and feed-forward layers
- **LLMEncoder**: Multi-layer encoder with token and positional embeddings
- Supports attention masking for padding tokens
- Configurable parameters: vocabulary size, model dimension, number of layers/heads

### Decoder (`decoder.py`) 
- **DecoderBlock**: Causal self-attention block for autoregressive generation
- **LLMDecoder**: Complete decoder with text generation capabilities
- Includes `generate()` method for autoregressive text generation
- Uses causal masking to prevent attention to future tokens

### Model Utils (`model_utils.py`)
- **MultiHeadAttention**: Scaled dot-product attention mechanism
- Supports multiple attention heads with proper dimensionality scaling
- Optional attention masking for padding and causal attention

### Tokenizer (`tokenizer.py`)
- **Tokenizer**: Byte-Pair Encoding (BPE) implementation
- Special tokens: `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`
- Training on text corpora with configurable vocabulary size
- Save/load functionality for trained tokenizers

## Usage

```python
# Initialize components
vocab_size = 50000
encoder = LLMEncoder(vocab_size=vocab_size)
decoder = LLMDecoder(vocab_size=vocab_size)
tokenizer = Tokenizer(vocab_size=vocab_size)

# Train tokenizer
texts = ["Your training texts here"]
tokenizer.train(texts)

# Encode text
input_ids = tokenizer.encode("Hello world!")

# Generate text
generated = decoder.generate(input_ids.unsqueeze(0), max_length=50)
output_text = tokenizer.decode(generated[0])
```

## Architecture Details

- **Model Dimension**: 512 (configurable)
- **Attention Heads**: 8 (configurable) 
- **Feed-Forward Dimension**: 2048 (configurable)
- **Max Sequence Length**: 1024
- **Activation**: GELU
- **Normalization**: Layer normalization with residual connections

## Requirements

- PyTorch
- Python 3.7+