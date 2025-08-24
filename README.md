# MLResearch - Transformer Language Model Implementation

A PyTorch implementation of transformer-based language models with encoder-decoder architecture and custom tokenization.

## Project Structure

```
MLResearch/
├── models/               # Model architectures
│   ├── encoder.py       # Encoder components
│   ├── decoder.py       # Decoder components  
│   ├── encoder_decoder.py # Combined encoder-decoder
│   └── model_utils.py   # Attention mechanisms
├── train/               # Training utilities
│   ├── train.py        # Main training script
│   ├── config_decoder.json # Decoder training config
│   └── config_encoder_decoder.json # Encoder-decoder config
├── tokenizer.py        # BPE tokenizer implementation
└── device_utils.py     # Device management utilities
```

## Components

### Models (`models/`)
- **EncoderBlock/LLMEncoder**: Self-attention encoder with positional embeddings
- **DecoderBlock/LLMDecoder**: Causal decoder with text generation capabilities  
- **EncoderDecoderModel**: Cross-attention transformer for sequence-to-sequence tasks
- **MultiHeadAttention**: Scaled dot-product attention mechanism

### Training (`train/`)
- **train.py**: Configuration-based training script supporting both model types
- **config_*.json**: Example training configurations with hyperparameters

### Utilities
- **Tokenizer**: BPE implementation with special tokens (`<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`)
- **DeviceManager**: Automatic device selection (CUDA/MPS/CPU) and tensor management

## Usage

### Training Models
```bash
# Train decoder-only model
cd train
python train.py --config config_decoder.json

# Train encoder-decoder model  
python train.py --config config_encoder_decoder.json
```

### Using Models Programmatically
```python
from models import LLMDecoder, EncoderDecoderModel
from tokenizer import Tokenizer

# Initialize components
vocab_size = 50000
decoder = LLMDecoder(vocab_size=vocab_size)
encoder_decoder = EncoderDecoderModel(vocab_size=vocab_size)
tokenizer = Tokenizer(vocab_size=vocab_size)

# Train tokenizer
texts = ["Your training texts here"]
tokenizer.train(texts)

# Generate text with decoder
input_ids = tokenizer.encode("Hello world!")
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