import os
import json
import requests
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from urllib.parse import urljoin
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextProcessor:
    """Process and clean raw text data."""
    
    def __init__(self, min_sentence_length: int = 20, max_sentence_length: int = 1000):
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove weird characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\'\"\-\(\)\[\]]', '', text)
        
        # Fix common issues
        text = text.replace('--', '—')  # Em dash
        text = text.replace(' ,', ',')   # Fix spacing
        text = text.replace(' .', '.')   # Fix spacing
        text = text.replace(' !', '!')   # Fix spacing
        text = text.replace(' ?', '?')   # Fix spacing
        
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with quality filtering."""
        # Basic sentence splitting
        sentences = re.split(r'[.!?]+', text)
        
        filtered_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Filter by length
            if len(sentence) < self.min_sentence_length or len(sentence) > self.max_sentence_length:
                continue
            
            # Filter out sentences that are mostly numbers or special characters
            if len(re.findall(r'\w', sentence)) < len(sentence) * 0.7:
                continue
            
            # Filter out sentences with too many uppercase letters (likely headers/titles)
            if len(re.findall(r'[A-Z]', sentence)) > len(sentence) * 0.3:
                continue
            
            filtered_sentences.append(sentence)
        
        return filtered_sentences
    
    def create_paragraphs(self, text: str, target_length: int = 200) -> List[str]:
        """Create coherent paragraphs of target length."""
        sentences = self.split_into_sentences(text)
        paragraphs = []
        current_paragraph = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) <= target_length:
                current_paragraph.append(sentence)
                current_length += len(sentence)
            else:
                if current_paragraph:
                    paragraphs.append('. '.join(current_paragraph) + '.')
                current_paragraph = [sentence]
                current_length = len(sentence)
        
        # Add the last paragraph
        if current_paragraph:
            paragraphs.append('. '.join(current_paragraph) + '.')
        
        return paragraphs


class GutenbergDownloader:
    """Download texts from Project Gutenberg."""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.base_url = "https://www.gutenberg.org/files/"
    
    def download_text(self, book_id: int, title: str = None) -> Optional[str]:
        """Download a book from Project Gutenberg."""
        cache_file = self.cache_dir / f"{book_id}.txt"
        
        # Check cache first
        if cache_file.exists():
            logger.info(f"Loading {title or book_id} from cache")
            return cache_file.read_text(encoding='utf-8', errors='ignore')
        
        # Download from Gutenberg
        url = f"{self.base_url}{book_id}/{book_id}-0.txt"
        try:
            logger.info(f"Downloading {title or book_id} from {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            text = response.text
            
            # Cache the downloaded text
            cache_file.write_text(text, encoding='utf-8')
            
            # Be respectful to the server
            time.sleep(1)
            
            return text
            
        except Exception as e:
            logger.error(f"Failed to download book {book_id}: {e}")
            return None
    
    def extract_main_text(self, raw_text: str) -> str:
        """Extract main text content from Gutenberg format."""
        lines = raw_text.split('\n')
        
        # Find start and end markers
        start_idx = 0
        end_idx = len(lines)
        
        for i, line in enumerate(lines):
            if "*** START OF THE PROJECT GUTENBERG EBOOK" in line.upper():
                start_idx = i + 1
                break
            elif "*** START OF THIS PROJECT GUTENBERG EBOOK" in line.upper():
                start_idx = i + 1
                break
        
        for i in range(len(lines) - 1, -1, -1):
            if "*** END OF THE PROJECT GUTENBERG EBOOK" in lines[i].upper():
                end_idx = i
                break
            elif "*** END OF THIS PROJECT GUTENBERG EBOOK" in lines[i].upper():
                end_idx = i
                break
        
        # Extract main text
        main_text = '\n'.join(lines[start_idx:end_idx])
        
        # Remove chapter headers and other formatting
        main_text = re.sub(r'^CHAPTER [IVXLC\d]+.*$', '', main_text, flags=re.MULTILINE)
        main_text = re.sub(r'^\s*\d+\s*$', '', main_text, flags=re.MULTILINE)  # Page numbers
        
        return main_text


class DatasetGenerator:
    """Generate training datasets from literary texts."""
    
    # High-quality literary works from Project Gutenberg
    QUALITY_BOOKS = {
        # Shakespeare
        1533: "The Tragedy of Hamlet, Prince of Denmark",
        1134: "The Tragedy of Romeo and Juliet", 
        1532: "King Lear",
        1135: "Othello",
        1130: "The Tempest",
        
        # Classic Literature
        1342: "Pride and Prejudice by Jane Austen",
        74: "The Adventures of Tom Sawyer by Mark Twain",
        76: "Adventures of Huckleberry Finn by Mark Twain",
        11: "Alice's Adventures in Wonderland by Lewis Carroll",
        174: "The Picture of Dorian Gray by Oscar Wilde",
        
        # Science Fiction Classics
        84: "Frankenstein by Mary Wollstonecraft Shelley",
        35: "The Time Machine by H. G. Wells",
        36: "The War of the Worlds by H. G. Wells",
        
        # Adventure/Mystery
        345: "Dracula by Bram Stoker",
        1661: "The Adventures of Sherlock Holmes by Arthur Conan Doyle",
        
        # Philosophy/Essays
        1232: "The Prince by Niccolò Machiavelli",
        2701: "Moby Dick by Herman Melville",
        
        # Poetry Collections
        1077: "The Complete Poems of Emily Dickinson",
        1364: "Leaves of Grass by Walt Whitman"
    }
    
    def __init__(self, output_dir: str = "generated_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.downloader = GutenbergDownloader()
        self.processor = TextProcessor()
    
    def generate_decoder_dataset(
        self, 
        book_ids: List[int] = None, 
        max_books: int = 5,
        output_file: str = "decoder_dataset.json"
    ) -> Dict:
        """Generate dataset for decoder-only language models."""
        if book_ids is None:
            book_ids = list(self.QUALITY_BOOKS.keys())[:max_books]
        
        all_texts = []
        metadata = {"sources": [], "total_texts": 0, "total_chars": 0}
        
        for book_id in book_ids:
            title = self.QUALITY_BOOKS.get(book_id, f"Book {book_id}")
            logger.info(f"Processing: {title}")
            
            # Download and process text
            raw_text = self.downloader.download_text(book_id, title)
            if not raw_text:
                continue
            
            # Extract main content
            main_text = self.downloader.extract_main_text(raw_text)
            clean_text = self.processor.clean_text(main_text)
            
            # Create paragraphs for training
            paragraphs = self.processor.create_paragraphs(clean_text, target_length=150)
            
            # Filter high-quality paragraphs
            quality_paragraphs = [p for p in paragraphs if len(p.split()) >= 20]
            
            all_texts.extend(quality_paragraphs)
            
            metadata["sources"].append({
                "book_id": book_id,
                "title": title,
                "paragraphs": len(quality_paragraphs),
                "total_chars": sum(len(p) for p in quality_paragraphs)
            })
        
        metadata["total_texts"] = len(all_texts)
        metadata["total_chars"] = sum(len(t) for t in all_texts)
        
        # Create dataset
        dataset = {
            "texts": all_texts,
            "metadata": metadata
        }
        
        # Save dataset
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Decoder dataset saved to {output_path}")
        logger.info(f"Total texts: {len(all_texts)}")
        logger.info(f"Total characters: {metadata['total_chars']:,}")
        
        return dataset
    
    def generate_translation_dataset(
        self,
        book_ids: List[int] = None,
        max_books: int = 3,
        output_file: str = "translation_dataset.json"
    ) -> Dict:
        """Generate synthetic translation dataset using paraphrasing."""
        if book_ids is None:
            book_ids = list(self.QUALITY_BOOKS.keys())[:max_books]
        
        source_texts = []
        target_texts = []
        metadata = {"sources": [], "total_pairs": 0}
        
        for book_id in book_ids:
            title = self.QUALITY_BOOKS.get(book_id, f"Book {book_id}")
            logger.info(f"Processing for translation pairs: {title}")
            
            # Download and process text
            raw_text = self.downloader.download_text(book_id, title)
            if not raw_text:
                continue
            
            main_text = self.downloader.extract_main_text(raw_text)
            clean_text = self.processor.clean_text(main_text)
            
            # Get sentences for paraphrasing
            sentences = self.processor.split_into_sentences(clean_text)
            quality_sentences = [s for s in sentences if 30 <= len(s.split()) <= 100]
            
            # Create simple paraphrases (this is a simplified version)
            # In practice, you might use more sophisticated paraphrasing
            for sentence in quality_sentences[:min(len(quality_sentences), 500)]:
                # Simple transformations for demonstration
                paraphrased = self._simple_paraphrase(sentence)
                if paraphrased and paraphrased != sentence:
                    source_texts.append(sentence)
                    target_texts.append(paraphrased)
            
            metadata["sources"].append({
                "book_id": book_id,
                "title": title,
                "sentence_pairs": len([s for s in quality_sentences[:500] if s])
            })
        
        metadata["total_pairs"] = len(source_texts)
        
        # Create dataset
        dataset = {
            "source": source_texts,
            "target": target_texts,
            "metadata": metadata
        }
        
        # Save dataset
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Translation dataset saved to {output_path}")
        logger.info(f"Total pairs: {len(source_texts)}")
        
        return dataset
    
    def _simple_paraphrase(self, sentence: str) -> Optional[str]:
        """Simple paraphrasing transformations."""
        # This is a very basic implementation
        # In practice, you'd want more sophisticated paraphrasing
        
        transformations = [
            (r'\bsaid\b', 'stated'),
            (r'\bvery\b', 'extremely'),
            (r'\bbig\b', 'large'),
            (r'\bsmall\b', 'little'),
            (r'\bgood\b', 'excellent'),
            (r'\bbad\b', 'terrible'),
            (r'\bhappy\b', 'joyful'),
            (r'\bsad\b', 'sorrowful'),
        ]
        
        paraphrased = sentence
        changed = False
        
        for pattern, replacement in transformations:
            if re.search(pattern, paraphrased, re.IGNORECASE):
                paraphrased = re.sub(pattern, replacement, paraphrased, count=1, flags=re.IGNORECASE)
                changed = True
                break
        
        return paraphrased if changed else None
    
    def generate_custom_dataset(
        self, 
        config_file: str
    ) -> Dict:
        """Generate dataset from custom configuration."""
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        dataset_type = config.get('type', 'decoder')
        book_selection = config.get('books', 'all')
        max_books = config.get('max_books', 5)
        output_file = config.get('output_file', 'custom_dataset.json')
        
        if book_selection == 'all':
            book_ids = list(self.QUALITY_BOOKS.keys())
        elif book_selection == 'shakespeare':
            book_ids = [1533, 1134, 1532, 1135, 1130]  # Shakespeare plays
        elif book_selection == 'classics':
            book_ids = [1342, 74, 76, 11, 174, 2701]   # Classic literature
        else:
            book_ids = book_selection  # Assume it's a list of IDs
        
        if dataset_type == 'decoder':
            return self.generate_decoder_dataset(book_ids[:max_books], output_file=output_file)
        elif dataset_type == 'translation':
            return self.generate_translation_dataset(book_ids[:max_books], output_file=output_file)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")


def main():
    parser = argparse.ArgumentParser(description='Generate high-quality training datasets from literature')
    parser.add_argument('--type', choices=['decoder', 'translation', 'custom'], 
                       default='decoder', help='Type of dataset to generate')
    parser.add_argument('--books', type=int, default=5, 
                       help='Maximum number of books to process')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file name')
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration file for custom generation')
    parser.add_argument('--output-dir', type=str, default='generated_data',
                       help='Output directory')
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(output_dir=args.output_dir)
    
    if args.type == 'custom' and args.config:
        dataset = generator.generate_custom_dataset(args.config)
    elif args.type == 'decoder':
        output_file = args.output or 'decoder_dataset.json'
        dataset = generator.generate_decoder_dataset(max_books=args.books, output_file=output_file)
    elif args.type == 'translation':
        output_file = args.output or 'translation_dataset.json'
        dataset = generator.generate_translation_dataset(max_books=args.books, output_file=output_file)
    
    logger.info("Dataset generation completed!")


if __name__ == "__main__":
    main()