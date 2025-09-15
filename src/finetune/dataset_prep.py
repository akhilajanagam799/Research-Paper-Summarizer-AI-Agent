"""
Dataset preparation for LoRA fine-tuning on research paper analysis tasks.

This module handles loading, processing, and splitting datasets for training
the paper analysis models with LoRA.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse
from datasets import Dataset
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetPreprocessor:
    """Handles dataset preparation for LoRA fine-tuning."""
    
    def __init__(self, tokenizer_name: str = "gpt2", max_length: int = 2048):
        """
        Initialize dataset preprocessor.
        
        Args:
            tokenizer_name: Name of tokenizer to use for length estimation
            max_length: Maximum sequence length
        """
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        
        # Try to load tokenizer for length estimation
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except:
            logger.warning("Could not load tokenizer, using word-based length estimation")
            self.tokenizer = None
    
    def load_jsonl_data(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """
        Load data from JSONL file.
        
        Args:
            jsonl_path: Path to JSONL file
            
        Returns:
            List of data records
        """
        if not Path(jsonl_path).exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
        
        data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    data.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
        
        logger.info(f"Loaded {len(data)} records from {jsonl_path}")
        return data
    
    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate a single data record.
        
        Args:
            record: Data record to validate
            
        Returns:
            (is_valid, error_message)
        """
        required_fields = ['instruction', 'input', 'response']
        
        for field in required_fields:
            if field not in record:
                return False, f"Missing required field: {field}"
            
            if not isinstance(record[field], str):
                return False, f"Field {field} must be a string"
            
            if not record[field].strip():
                return False, f"Field {field} cannot be empty"
        
        # Check length constraints
        if self.tokenizer:
            full_text = f"{record['instruction']} {record['input']} {record['response']}"
            tokens = self.tokenizer(full_text, return_tensors="pt", truncation=False)
            token_count = tokens['input_ids'].shape[1]
            
            if token_count > self.max_length:
                return False, f"Record too long: {token_count} tokens (max: {self.max_length})"
        else:
            # Approximate token count (1 token ≈ 0.75 words)
            full_text = f"{record['instruction']} {record['input']} {record['response']}"
            word_count = len(full_text.split())
            estimated_tokens = int(word_count * 1.33)
            
            if estimated_tokens > self.max_length:
                return False, f"Record too long: ~{estimated_tokens} tokens (max: {self.max_length})"
        
        return True, ""
    
    def validate_dataset(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Validate entire dataset and filter invalid records.
        
        Args:
            data: List of data records
            
        Returns:
            (valid_records, error_messages)
        """
        valid_records = []
        errors = []
        
        for i, record in enumerate(data):
            is_valid, error = self.validate_record(record)
            
            if is_valid:
                valid_records.append(record)
            else:
                errors.append(f"Record {i}: {error}")
        
        logger.info(f"Validated dataset: {len(valid_records)}/{len(data)} records valid")
        
        if errors:
            logger.warning(f"Found {len(errors)} invalid records")
            for error in errors[:5]:  # Show first 5 errors
                logger.warning(error)
            if len(errors) > 5:
                logger.warning(f"... and {len(errors) - 5} more errors")
        
        return valid_records, errors
    
    def format_training_data(self, records: List[Dict[str, Any]], format_type: str = "alpaca") -> List[Dict[str, str]]:
        """
        Format data for training.
        
        Args:
            records: Validated data records
            format_type: Training format ("alpaca", "chat", "completion")
            
        Returns:
            Formatted training data
        """
        formatted_data = []
        
        for record in records:
            if format_type == "alpaca":
                # Alpaca format: instruction + input -> response
                if record['input'].strip():
                    prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{record['instruction']}\n\n### Input:\n{record['input']}\n\n### Response:\n"
                else:
                    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{record['instruction']}\n\n### Response:\n"
                
                formatted_record = {
                    "prompt": prompt,
                    "response": record['response']
                }
                
            elif format_type == "chat":
                # Chat format with system/user/assistant roles
                formatted_record = {
                    "messages": [
                        {"role": "system", "content": "You are an expert researcher analyzing academic papers."},
                        {"role": "user", "content": f"{record['instruction']}\n\n{record['input']}"},
                        {"role": "assistant", "content": record['response']}
                    ]
                }
                
            elif format_type == "completion":
                # Simple completion format
                formatted_record = {
                    "text": f"{record['instruction']}\n\nContext: {record['input']}\n\nResponse: {record['response']}<|endoftext|>"
                }
                
            else:
                raise ValueError(f"Unknown format type: {format_type}")
            
            formatted_data.append(formatted_record)
        
        logger.info(f"Formatted {len(formatted_data)} records in {format_type} format")
        return formatted_data
    
    def split_dataset(
        self,
        data: List[Dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify_by: str = None,
        random_seed: int = 42
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            data: Data to split
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            stratify_by: Field to stratify by (e.g., 'task_type')
            random_seed: Random seed
            
        Returns:
            (train_data, val_data, test_data)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        random.seed(random_seed)
        
        if stratify_by and stratify_by in data[0]:
            # Stratified split
            grouped_data = {}
            for record in data:
                key = record[stratify_by]
                if key not in grouped_data:
                    grouped_data[key] = []
                grouped_data[key].append(record)
            
            train_data, val_data, test_data = [], [], []
            
            for key, group in grouped_data.items():
                random.shuffle(group)
                
                n = len(group)
                n_train = int(n * train_ratio)
                n_val = int(n * val_ratio)
                
                train_data.extend(group[:n_train])
                val_data.extend(group[n_train:n_train + n_val])
                test_data.extend(group[n_train + n_val:])
            
            logger.info(f"Stratified split by {stratify_by}")
            
        else:
            # Random split
            data_copy = data.copy()
            random.shuffle(data_copy)
            
            n = len(data_copy)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            train_data = data_copy[:n_train]
            val_data = data_copy[n_train:n_train + n_val]
            test_data = data_copy[n_train + n_val:]
        
        logger.info(f"Dataset split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        return train_data, val_data, test_data
    
    def save_datasets(
        self,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]],
        output_dir: str,
        format_type: str = "alpaca"
    ):
        """
        Save datasets to files.
        
        Args:
            train_data: Training data
            val_data: Validation data  
            test_data: Test data
            output_dir: Output directory
            format_type: Data format
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Format data
        train_formatted = self.format_training_data(train_data, format_type)
        val_formatted = self.format_training_data(val_data, format_type)
        test_formatted = self.format_training_data(test_data, format_type)
        
        # Save as JSONL
        datasets = {
            'train': train_formatted,
            'validation': val_formatted,
            'test': test_formatted
        }
        
        for split, data in datasets.items():
            filepath = output_path / f"{split}.jsonl"
            with open(filepath, 'w', encoding='utf-8') as f:
                for record in data:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            logger.info(f"Saved {len(data)} {split} records to {filepath}")
        
        # Save metadata
        metadata = {
            'total_records': len(train_data) + len(val_data) + len(test_data),
            'splits': {
                'train': len(train_data),
                'validation': len(val_data),
                'test': len(test_data)
            },
            'format': format_type,
            'max_length': self.max_length,
            'tokenizer': self.tokenizer_name
        }
        
        metadata_path = output_path / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")
    
    def create_hf_datasets(self, data_dir: str) -> Dict[str, Dataset]:
        """
        Create HuggingFace datasets from saved files.
        
        Args:
            data_dir: Directory containing dataset files
            
        Returns:
            Dictionary of datasets
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        datasets = {}
        
        for split in ['train', 'validation', 'test']:
            file_path = data_path / f"{split}.jsonl"
            
            if file_path.exists():
                # Load JSONL data
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                
                # Convert to HuggingFace dataset
                datasets[split] = Dataset.from_list(data)
                logger.info(f"Created {split} dataset with {len(data)} records")
        
        return datasets
    
    def analyze_dataset(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze dataset statistics.
        
        Args:
            data: Dataset to analyze
            
        Returns:
            Analysis results
        """
        if not data:
            return {"error": "Empty dataset"}
        
        analysis = {
            'total_records': len(data),
            'task_distribution': {},
            'length_stats': {
                'instruction': [],
                'input': [],
                'response': [],
                'total': []
            }
        }
        
        # Analyze task distribution
        for record in data:
            instruction = record.get('instruction', '')
            
            # Categorize by task type keywords
            task_type = 'other'
            if 'summarize' in instruction.lower() or 'summary' in instruction.lower():
                task_type = 'summary'
            elif 'glossary' in instruction.lower() or 'terms' in instruction.lower():
                task_type = 'glossary'  
            elif 'question' in instruction.lower():
                task_type = 'questions'
            
            analysis['task_distribution'][task_type] = analysis['task_distribution'].get(task_type, 0) + 1
            
            # Length analysis
            if self.tokenizer:
                for field in ['instruction', 'input', 'response']:
                    text = record.get(field, '')
                    tokens = self.tokenizer(text, return_tensors="pt", truncation=False)
                    length = tokens['input_ids'].shape[1]
                    analysis['length_stats'][field].append(length)
                
                total_text = f"{record.get('instruction', '')} {record.get('input', '')} {record.get('response', '')}"
                total_tokens = self.tokenizer(total_text, return_tensors="pt", truncation=False)
                analysis['length_stats']['total'].append(total_tokens['input_ids'].shape[1])
            else:
                # Word-based approximation
                for field in ['instruction', 'input', 'response']:
                    text = record.get(field, '')
                    words = len(text.split())
                    analysis['length_stats'][field].append(int(words * 1.33))  # Approximate tokens
                
                total_text = f"{record.get('instruction', '')} {record.get('input', '')} {record.get('response', '')}"
                total_words = len(total_text.split())
                analysis['length_stats']['total'].append(int(total_words * 1.33))
        
        # Calculate statistics
        for field in analysis['length_stats']:
            lengths = analysis['length_stats'][field]
            if lengths:
                analysis['length_stats'][field] = {
                    'min': min(lengths),
                    'max': max(lengths),
                    'mean': sum(lengths) / len(lengths),
                    'median': sorted(lengths)[len(lengths) // 2]
                }
        
        return analysis


def main():
    """CLI for dataset preparation."""
    parser = argparse.ArgumentParser(description="Prepare dataset for LoRA fine-tuning")
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help="Path to input JSONL file"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/processed',
        help="Output directory for processed datasets"
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['alpaca', 'chat', 'completion'],
        default='alpaca',
        help="Training data format"
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help="Training set ratio"
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help="Validation set ratio"
    )
    
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help="Test set ratio"
    )
    
    parser.add_argument(
        '--stratify',
        type=str,
        help="Field to stratify split by"
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='gpt2',
        help="Tokenizer for length estimation"
    )
    
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help="Only analyze dataset, don't process"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize preprocessor
        preprocessor = DatasetPreprocessor(
            tokenizer_name=args.tokenizer,
            max_length=args.max_length
        )
        
        # Load data
        data = preprocessor.load_jsonl_data(args.input)
        
        # Validate data
        valid_data, errors = preprocessor.validate_dataset(data)
        
        if not valid_data:
            logger.error("No valid records found")
            return
        
        # Analyze dataset
        analysis = preprocessor.analyze_dataset(valid_data)
        
        print("\n" + "="*50)
        print("DATASET ANALYSIS")
        print("="*50)
        
        print(f"Total records: {analysis['total_records']}")
        print(f"Task distribution: {analysis['task_distribution']}")
        
        for field, stats in analysis['length_stats'].items():
            if isinstance(stats, dict):
                print(f"{field.capitalize()} length - Min: {stats['min']}, Max: {stats['max']}, Mean: {stats['mean']:.1f}")
        
        if args.analyze_only:
            return
        
        # Split dataset
        train_data, val_data, test_data = preprocessor.split_dataset(
            valid_data,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            stratify_by=args.stratify
        )
        
        # Save datasets
        preprocessor.save_datasets(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            output_dir=args.output,
            format_type=args.format
        )
        
        print(f"\nDatasets saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise


if __name__ == "__main__":
    main()