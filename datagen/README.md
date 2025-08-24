# Data Generation for Language Model Training

Generate high-quality training datasets from classic literature via Project Gutenberg.

## Features

- **Quality Sources**: Curated selection from Shakespeare, Austen, Twain, Carroll, Wilde, and more
- **Text Processing**: Intelligent cleaning, sentence splitting, and paragraph formation
- **Multiple Formats**: Support for decoder-only and encoder-decoder training data
- **Caching**: Downloaded texts are cached to avoid re-downloading
- **Configurable**: Flexible configuration for different dataset requirements

## Available Literature

### Shakespeare Collection
- Hamlet (1533)
- Romeo and Juliet (1134) 
- King Lear (1532)
- Othello (1135)
- The Tempest (1130)

### Classic Literature
- Pride and Prejudice - Jane Austen (1342)
- Tom Sawyer - Mark Twain (74)
- Huckleberry Finn - Mark Twain (76)
- Alice in Wonderland - Lewis Carroll (11)
- Dorian Gray - Oscar Wilde (174)
- Moby Dick - Herman Melville (2701)

### Science Fiction Classics
- Frankenstein - Mary Shelley (84)
- The Time Machine - H.G. Wells (35)
- War of the Worlds - H.G. Wells (36)

### Adventure/Mystery
- Dracula - Bram Stoker (345)
- Sherlock Holmes - Arthur Conan Doyle (1661)

## Usage

### Basic Generation
```bash
# Generate decoder dataset from 5 books
python datagen.py --type decoder --books 5

# Generate translation dataset (with paraphrasing)
python datagen.py --type translation --books 3

# Custom output file
python datagen.py --type decoder --books 3 --output my_dataset.json
```

### Configuration-Based Generation
```bash
# Use predefined configurations
python datagen.py --type custom --config shakespeare_config.json
python datagen.py --type custom --config classics_config.json
```

### Custom Configuration
Create a JSON config file:
```json
{
  "type": "decoder",
  "books": [1342, 74, 11],
  "max_books": 3,
  "output_file": "custom_dataset.json",
  "preprocessing": {
    "min_sentence_length": 30,
    "max_sentence_length": 800,
    "target_paragraph_length": 200
  }
}
```

## Output Formats

### Decoder Dataset
```json
{
  "texts": [
    "It was the best of times, it was the worst of times...",
    "To be or not to be, that is the question...",
    ...
  ],
  "metadata": {
    "sources": [...],
    "total_texts": 1250,
    "total_chars": 45000
  }
}
```

### Translation Dataset
```json
{
  "source": ["Original sentences..."],
  "target": ["Paraphrased versions..."],
  "metadata": {...}
}
```

## Text Processing Features

- **Intelligent Cleaning**: Removes formatting artifacts while preserving literary style
- **Quality Filtering**: Filters out headers, page numbers, and low-quality sentences
- **Sentence Segmentation**: Proper sentence boundary detection
- **Paragraph Formation**: Creates coherent paragraphs of target length
- **Length Filtering**: Configurable minimum/maximum lengths

## Requirements

```bash
pip install requests
```

## File Structure

```
datagen/
├── datagen.py              # Main generation script
├── shakespeare_config.json # Shakespeare-specific config
├── classics_config.json    # Classic literature config
├── cache/                  # Downloaded text cache
└── generated_data/         # Output datasets
```

## Notes

- Respects Project Gutenberg's servers with rate limiting
- All texts are from public domain sources
- Caches downloads to avoid repeated requests
- Processes texts to remove Gutenberg headers/footers
- Maintains literary quality while creating training-suitable chunks