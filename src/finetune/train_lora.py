"""
LoRA fine-tuning script for research paper analysis models.

This script handles the complete LoRA training pipeline using PEFT and
the HuggingFace Trainer API with support for distributed training.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from datasets import Dataset, load_dataset
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoRATrainer:
    """Handles LoRA fine-tuning for language models."""
    
    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        max_length: int = 2048,
        use_4bit: bool = True,
        use_8bit: bool = False,
        trust_remote_code: bool = True
    ):
        """
        Initialize LoRA trainer.
        
        Args:
            model_name: Base model name
            max_length: Maximum sequence length
            use_4bit: Use 4-bit quantization
            use_8bit: Use 8-bit quantization
            trust_remote_code: Trust remote code for model loading
        """
        self.model_name = model_name
        self.max_length = max_length
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.trust_remote_code = trust_remote_code
        
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        if self.device == "cpu" and (use_4bit or use_8bit):
            logger.warning("Quantization not supported on CPU, disabling")
            self.use_4bit = False
            self.use_8bit = False
    
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Setup quantization config
        quantization_config = None
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_quant_storage=torch.uint8
            )
        elif self.use_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            padding_side="left"
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Move to device if not using quantization
        if not quantization_config and self.device == "cuda":
            self.model = self.model.to(self.device)
        
        logger.info("Model and tokenizer loaded successfully")
    
    def setup_lora_config(
        self,
        r: int = 8,
        alpha: int = 32,
        dropout: float = 0.05,
        target_modules: Optional[list] = None
    ) -> LoraConfig:
        """
        Setup LoRA configuration.
        
        Args:
            r: LoRA rank
            alpha: LoRA alpha parameter
            dropout: LoRA dropout
            target_modules: Target modules for LoRA
            
        Returns:
            LoRA configuration
        """
        if target_modules is None:
            # Default target modules for common architectures
            if "llama" in self.model_name.lower():
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "phi" in self.model_name.lower():
                target_modules = ["Wqkv", "out_proj", "fc1", "fc2"]
            elif "gpt" in self.model_name.lower():
                target_modules = ["c_attn", "c_proj", "c_fc"]
            else:
                # Generic attention modules
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        logger.info(f"LoRA config: r={r}, alpha={alpha}, dropout={dropout}")
        logger.info(f"Target modules: {target_modules}")
        
        return lora_config
    
    def prepare_model_for_training(self, lora_config: LoraConfig):
        """Prepare model with LoRA for training."""
        logger.info("Setting up LoRA model for training")
        
        # Apply LoRA
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = 0
        all_param = 0
        for _, param in self.peft_model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable%: {100 * trainable_params / all_param:.2f}%"
        )
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.peft_model, 'gradient_checkpointing_enable'):
            self.peft_model.gradient_checkpointing_enable()
        
        return self.peft_model
    
    def load_datasets(self, data_dir: str) -> Dict[str, Dataset]:
        """
        Load training datasets.
        
        Args:
            data_dir: Directory containing dataset files
            
        Returns:
            Dictionary of datasets
        """
        logger.info(f"Loading datasets from {data_dir}")
        
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        datasets = {}
        
        for split in ['train', 'validation']:
            file_path = data_path / f"{split}.jsonl"
            
            if file_path.exists():
                dataset = load_dataset('json', data_files=str(file_path), split='train')
                datasets[split] = dataset
                logger.info(f"Loaded {split} dataset: {len(dataset)} examples")
        
        if 'train' not in datasets:
            raise FileNotFoundError("Training dataset not found")
        
        return datasets
    
    def preprocess_function(self, examples):
        """Preprocess examples for training."""
        # Handle different formats
        if 'prompt' in examples and 'response' in examples:
            # Alpaca format
            inputs = [f"{prompt}{response}" for prompt, response in zip(examples['prompt'], examples['response'])]
        elif 'text' in examples:
            # Completion format
            inputs = examples['text']
        else:
            # Fallback
            inputs = [f"{ex.get('instruction', '')}\n{ex.get('input', '')}\n{ex.get('response', '')}" 
                     for ex in examples]
        
        # Tokenize
        model_inputs = self.tokenizer(
            inputs,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None
        )
        
        # Set labels equal to input_ids for causal LM
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs
    
    def train(
        self,
        datasets: Dict[str, Dataset],
        output_dir: str = "models/lora-adapter",
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        per_device_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: int = 500,
        lora_config: Optional[LoraConfig] = None,
        use_wandb: bool = False,
        wandb_project: str = "research-paper-summarizer"
    ):
        """
        Train the LoRA model.
        
        Args:
            datasets: Training datasets
            output_dir: Output directory for checkpoints
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            per_device_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Warmup steps
            logging_steps: Logging frequency
            save_steps: Save frequency
            eval_steps: Evaluation frequency
            lora_config: LoRA configuration
            use_wandb: Use Weights & Biases for logging
            wandb_project: W&B project name
        """
        # Setup LoRA if not provided
        if lora_config is None:
            lora_config = self.setup_lora_config()
        
        # Prepare model
        model = self.prepare_model_for_training(lora_config)
        
        # Preprocess datasets
        train_dataset = datasets['train'].map(
            self.preprocess_function,
            batched=True,
            remove_columns=datasets['train'].column_names
        )
        
        eval_dataset = None
        if 'validation' in datasets:
            eval_dataset = datasets['validation'].map(
                self.preprocess_function,
                batched=True,
                remove_columns=datasets['validation'].column_names
            )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            report_to="wandb" if use_wandb else None,
            run_name=f"lora-{self.model_name.split('/')[-1]}" if use_wandb else None,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            optim="adamw_torch",
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        )
        
        # Initialize W&B if requested
        if use_wandb:
            wandb.init(
                project=wandb_project,
                config={
                    "model_name": self.model_name,
                    "lora_r": lora_config.r,
                    "lora_alpha": lora_config.lora_alpha,
                    "learning_rate": learning_rate,
                    "batch_size": per_device_batch_size,
                    "epochs": num_epochs
                }
            )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model()
        
        # Save training metrics
        training_results = {
            "model_name": self.model_name,
            "lora_config": {
                "r": lora_config.r,
                "alpha": lora_config.lora_alpha,
                "dropout": lora_config.lora_dropout,
                "target_modules": lora_config.target_modules
            },
            "training_args": training_args.to_dict(),
            "train_dataset_size": len(train_dataset),
            "eval_dataset_size": len(eval_dataset) if eval_dataset else 0,
            "final_loss": trainer.state.log_history[-1].get('train_loss', 0),
            "final_eval_loss": trainer.state.log_history[-1].get('eval_loss', 0) if eval_dataset else None
        }
        
        results_path = Path(output_dir) / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2)
        
        logger.info("Training completed successfully!")
        
        if use_wandb:
            wandb.finish()
        
        return trainer, training_results


def main():
    """CLI for LoRA training."""
    parser = argparse.ArgumentParser(description="Train LoRA adapter for research paper analysis")
    
    # Model arguments
    parser.add_argument(
        '--model-name', '-m',
        type=str,
        default='microsoft/phi-2',
        help='Base model name (default: microsoft/phi-2)'
    )
    
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default='data/processed',
        help='Data directory containing train.jsonl and validation.jsonl'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='models/lora-adapter',
        help='Output directory for LoRA adapter'
    )
    
    # LoRA arguments
    parser.add_argument('--lora-r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=1, help='Per-device batch size')
    parser.add_argument('--gradient-accumulation', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--max-length', type=int, default=2048, help='Maximum sequence length')
    
    # System arguments
    parser.add_argument('--no-4bit', action='store_true', help='Disable 4-bit quantization')
    parser.add_argument('--use-8bit', action='store_true', help='Use 8-bit quantization instead')
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU training')
    
    # Logging arguments
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='research-paper-summarizer', 
                       help='W&B project name')
    
    args = parser.parse_args()
    
    try:
        # Override device if CPU-only
        if args.cpu_only:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        # Initialize trainer
        trainer = LoRATrainer(
            model_name=args.model_name,
            max_length=args.max_length,
            use_4bit=not args.no_4bit and not args.cpu_only,
            use_8bit=args.use_8bit and not args.cpu_only
        )
        
        # Load model and tokenizer
        trainer.load_model_and_tokenizer()
        
        # Load datasets
        datasets = trainer.load_datasets(args.data_dir)
        
        # Setup LoRA config
        lora_config = trainer.setup_lora_config(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout
        )
        
        # Train
        trained_model, results = trainer.train(
            datasets=datasets,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            per_device_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            lora_config=lora_config,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project
        )
        
        print(f"\nTraining completed! Model saved to: {args.output_dir}")
        print(f"Final training loss: {results.get('final_loss', 'N/A')}")
        if results.get('final_eval_loss'):
            print(f"Final validation loss: {results['final_eval_loss']}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()