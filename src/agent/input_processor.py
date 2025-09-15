"""
PDF processing and text extraction module for research papers.

This module handles PDF extraction, text cleaning, and chunking operations
using PyMuPDF for robust document processing.
"""

import re
import fitz  # PyMuPDF
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF extraction and text preprocessing."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize PDF processor.
        
        Args:
            chunk_size: Maximum tokens per chunk
            overlap: Overlap tokens between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dict containing extracted text, metadata, and page info
        """
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        try:
            doc = fitz.open(pdf_path)
            
            pages = []
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                pages.append({
                    'page_number': page_num + 1,
                    'text': text,
                    'char_count': len(text)
                })
                
                full_text += text + "\n"
            
            doc.close()
            
            # Extract metadata
            metadata = self._extract_metadata(full_text)
            
            result = {
                'full_text': full_text,
                'pages': pages,
                'metadata': metadata,
                'total_pages': len(pages),
                'total_chars': len(full_text)
            }
            
            logger.info(f"Extracted text from {len(pages)} pages ({len(full_text)} characters)")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove page headers/footers (common patterns)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Page \d+.*$', '', text, flags=re.MULTILINE)
        
        # Fix common OCR issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Remove URLs and DOIs
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'doi:\S+', '', text)
        
        # Clean up spacing
        text = text.strip()
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text
    
    def chunk_text(self, text: str, method: str = 'token') -> List[Dict[str, any]]:
        """
        Chunk text into smaller segments for processing.
        
        Args:
            text: Cleaned text to chunk
            method: Chunking method ('token' or 'sentence')
            
        Returns:
            List of text chunks with metadata
        """
        if method == 'token':
            return self._chunk_by_tokens(text)
        elif method == 'sentence':
            return self._chunk_by_sentences(text)
        else:
            raise ValueError(f"Unknown chunking method: {method}")
    
    def _chunk_by_tokens(self, text: str) -> List[Dict[str, any]]:
        """Chunk text by approximate token count."""
        words = text.split()
        chunks = []
        
        i = 0
        chunk_id = 0
        
        while i < len(words):
            # Approximate tokens (1 token ≈ 0.75 words)
            chunk_word_limit = int(self.chunk_size * 0.75)
            overlap_words = int(self.overlap * 0.75)
            
            chunk_words = words[i:i + chunk_word_limit]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'start_word': i,
                'end_word': i + len(chunk_words),
                'word_count': len(chunk_words),
                'char_count': len(chunk_text)
            })
            
            # Move forward with overlap
            i += max(1, chunk_word_limit - overlap_words)
            chunk_id += 1
        
        logger.info(f"Created {len(chunks)} text chunks")
        return chunks
    
    def _chunk_by_sentences(self, text: str) -> List[Dict[str, any]]:
        """Chunk text by sentences within token limits."""
        sentences = re.split(r'[.!?]+\s+', text)
        chunks = []
        
        current_chunk = []
        current_length = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Approximate tokens
            sentence_length = len(sentence.split()) * 1.33  # words to tokens
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'sentence_count': len(current_chunk),
                    'word_count': len(chunk_text.split()),
                    'char_count': len(chunk_text)
                })
                
                # Start new chunk with overlap
                overlap_sentences = int(len(current_chunk) * self.overlap / self.chunk_size)
                current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
                current_length = sum(len(s.split()) * 1.33 for s in current_chunk)
                chunk_id += 1
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'sentence_count': len(current_chunk),
                'word_count': len(chunk_text.split()),
                'char_count': len(chunk_text)
            })
        
        logger.info(f"Created {len(chunks)} sentence-based chunks")
        return chunks
    
    def _extract_metadata(self, text: str) -> Dict[str, any]:
        """Extract paper metadata from text."""
        metadata = {
            'title': None,
            'authors': [],
            'abstract': None,
            'keywords': [],
            'sections': []
        }
        
        lines = text.split('\n')
        
        # Extract title (usually first substantial line)
        for line in lines[:10]:
            line = line.strip()
            if len(line) > 10 and not line.isupper() and not line.startswith(('Figure', 'Table')):
                metadata['title'] = line
                break
        
        # Find abstract
        abstract_start = -1
        for i, line in enumerate(lines):
            if re.search(r'\babstract\b', line.lower()):
                abstract_start = i
                break
        
        if abstract_start != -1:
            abstract_text = []
            for line in lines[abstract_start:abstract_start + 20]:
                line = line.strip()
                if line and not line.startswith(('Keywords', 'Introduction', '1.')):
                    abstract_text.append(line)
                elif abstract_text:
                    break
            
            if abstract_text:
                metadata['abstract'] = ' '.join(abstract_text[1:])  # Skip "Abstract" line
        
        # Find sections
        sections = []
        for line in lines:
            # Look for numbered sections
            section_match = re.match(r'^(\d+\.?\d*)\s+([A-Z][^.]{2,50})', line.strip())
            if section_match:
                sections.append({
                    'number': section_match.group(1),
                    'title': section_match.group(2).strip()
                })
        
        metadata['sections'] = sections
        
        return metadata
    
    def get_top_chunks(self, chunks: List[Dict], method: str = 'length', k: int = 5) -> List[Dict]:
        """
        Select top-K chunks for processing.
        
        Args:
            chunks: List of text chunks
            method: Selection method ('length', 'position', 'mixed')
            k: Number of chunks to return
            
        Returns:
            Top-K selected chunks
        """
        if method == 'length':
            # Select longest chunks
            sorted_chunks = sorted(chunks, key=lambda x: x['char_count'], reverse=True)
        elif method == 'position':
            # Select from beginning and middle
            indices = list(range(0, min(k//2, len(chunks))))
            middle_start = len(chunks) // 3
            indices.extend(range(middle_start, min(middle_start + k//2, len(chunks))))
            sorted_chunks = [chunks[i] for i in indices[:k]]
        elif method == 'mixed':
            # Mix of early chunks (introduction) and long chunks (content)
            early_chunks = chunks[:min(3, len(chunks))]
            long_chunks = sorted(chunks[3:], key=lambda x: x['char_count'], reverse=True)
            sorted_chunks = early_chunks + long_chunks[:max(0, k-3)]
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        return sorted_chunks[:k]


def process_paper(pdf_path: str, chunk_method: str = 'token', 
                 selection_method: str = 'mixed', top_k: int = 5) -> Dict[str, any]:
    """
    Complete paper processing pipeline.
    
    Args:
        pdf_path: Path to PDF file
        chunk_method: Chunking method
        selection_method: Chunk selection method
        top_k: Number of top chunks to return
        
    Returns:
        Processed paper data
    """
    processor = PDFProcessor()
    
    # Extract and clean text
    pdf_data = processor.extract_text_from_pdf(pdf_path)
    clean_text = processor.clean_text(pdf_data['full_text'])
    
    # Chunk text
    chunks = processor.chunk_text(clean_text, method=chunk_method)
    
    # Select top chunks
    top_chunks = processor.get_top_chunks(chunks, method=selection_method, k=top_k)
    
    result = {
        'metadata': pdf_data['metadata'],
        'statistics': {
            'total_pages': pdf_data['total_pages'],
            'total_chars': len(clean_text),
            'total_chunks': len(chunks),
            'selected_chunks': len(top_chunks)
        },
        'text': {
            'full_text': clean_text,
            'all_chunks': chunks,
            'selected_chunks': top_chunks
        }
    }
    
    return result


if __name__ == "__main__":
    # Test the processor
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python input_processor.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    result = process_paper(pdf_path)
    
    print(f"Processed paper: {result['metadata']['title']}")
    print(f"Pages: {result['statistics']['total_pages']}")
    print(f"Characters: {result['statistics']['total_chars']}")
    print(f"Chunks: {result['statistics']['total_chunks']}")
    print(f"Selected chunks: {result['statistics']['selected_chunks']}")
    
    # Save results
    output_path = "logs/processed_paper.json"
    Path(output_path).parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to: {output_path}")