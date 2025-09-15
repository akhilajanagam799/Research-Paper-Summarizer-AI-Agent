"""
Model execution module with LoRA support for research paper analysis.

This module handles model loading, LoRA adapter integration, and text generation
with support for both CPU and GPU inference.
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig,
    BitsAndBytesConfig
)
from peft import PeftModel, PeftConfig
import logging
from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExecutor:
    """Handles model loading and text generation with LoRA support."""
    
    # Default models for different scenarios
    DEFAULT_MODELS = {
        'cpu': 'gpt2',  # Fast CPU testing
        'gpu_small': 'microsoft/phi-2',  # GPU with moderate memory
        'gpu_large': 'meta-llama/Llama-2-7b-chat-hf'  # GPU with large memory
    }
    
    def __init__(
        self,
        model_name: str = 'gpt2',
        device: str = 'auto',
        use_8bit: bool = False,
        use_4bit: bool = False,
        lora_adapter_path: Optional[str] = None
    ):
        """
        Initialize model executor.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device for inference ('auto', 'cpu', 'cuda')
            use_8bit: Use 8-bit quantization (requires GPU)
            use_4bit: Use 4-bit quantization (requires GPU)
            lora_adapter_path: Path to LoRA adapter
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.use_8bit = use_8bit and self.device != 'cpu'
        self.use_4bit = use_4bit and self.device != 'cpu'
        self.lora_adapter_path = lora_adapter_path
        
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        
        logger.info(f"Initializing executor: {model_name} on {self.device}")
        self._load_model()
    
    def _setup_device(self, device: str) -> str:
        """Setup and validate device."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        
        logger.info(f"Using device: {device}")
        return device
    
    def _load_model(self):
        """Load model and tokenizer with optional quantization."""
        try:
            # Setup quantization config
            quantization_config = None
            if self.use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.use_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            logger.info("Loading model...")
            model_kwargs = {
                'trust_remote_code': True,
                'torch_dtype': torch.float16 if self.device != 'cpu' else torch.float32,
                'device_map': 'auto' if self.device == 'cuda' else None
            }
            
            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Load LoRA adapter if specified
            if self.lora_adapter_path and Path(self.lora_adapter_path).exists():
                logger.info(f"Loading LoRA adapter from {self.lora_adapter_path}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.lora_adapter_path,
                    torch_dtype=torch.float16 if self.device != 'cpu' else torch.float32
                )
            elif self.lora_adapter_path:
                logger.warning(f"LoRA adapter path not found: {self.lora_adapter_path}")
            
            # Move to device if needed
            if self.device == 'cpu' and not quantization_config:
                self.model = self.model.to(self.device)
            
            # Setup generation config
            self._setup_generation_config()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _setup_generation_config(self):
        """Setup generation configuration."""
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            length_penalty=1.0,
            early_stopping=True
        )
    
    def format_messages(self, system_prompt: str, user_prompt: str) -> str:
        """
        Format system and user prompts for the model.
        
        Args:
            system_prompt: System instruction
            user_prompt: User query
            
        Returns:
            Formatted prompt string
        """
        # Handle different model formats
        if 'llama' in self.model_name.lower():
            # Llama-2 chat format
            formatted = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
        elif 'phi' in self.model_name.lower():
            # Phi-2 format
            formatted = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        else:
            # Generic format for GPT-2 and others
            formatted = f"{system_prompt}\n\n{user_prompt}\n\nResponse:"
        
        return formatted
    
    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate text response.
        
        Args:
            system_prompt: System instruction
            user_prompt: User query
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text and metadata
        """
        start_time = time.time()
        
        try:
            # Format prompt
            formatted_prompt = self.format_messages(system_prompt, user_prompt)
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096 - max_tokens  # Leave room for generation
            )
            
            if self.device != 'cpu':
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Update generation config
            gen_config = GenerationConfig(
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                early_stopping=True
            )
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Decode response
            generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Clean up response
            response = response.strip()
            
            # Calculate metrics
            generation_time = time.time() - start_time
            input_tokens = inputs['input_ids'].shape[1]
            output_tokens = len(generated_ids)
            
            result = {
                'text': response,
                'metadata': {
                    'model_name': self.model_name,
                    'device': self.device,
                    'generation_time': generation_time,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens,
                    'tokens_per_second': output_tokens / generation_time if generation_time > 0 else 0,
                    'temperature': temperature,
                    'lora_adapter': self.lora_adapter_path is not None
                }
            }
            
            logger.info(f"Generated {output_tokens} tokens in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return {
                'text': f"Error generating response: {str(e)}",
                'metadata': {
                    'error': str(e),
                    'generation_time': time.time() - start_time
                }
            }
    
    def execute_task(self, task_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a planned analysis task.
        
        Args:
            task_plan: Task plan from planner
            
        Returns:
            Task execution results
        """
        start_time = time.time()
        task_type = task_plan['task_type']
        prompt = task_plan['prompt']
        
        logger.info(f"Executing {task_type} task")
        
        # Generate response
        result = self.generate_text(
            system_prompt=prompt['system'],
            user_prompt=prompt['user'],
            max_tokens=self._get_max_tokens_for_task(task_type),
            temperature=self._get_temperature_for_task(task_type)
        )
        
        # Post-process based on task type
        processed_output = self._post_process_output(task_type, result['text'])
        
        execution_result = {
            'task_type': task_type,
            'output': processed_output,
            'raw_output': result['text'],
            'metadata': {
                **result['metadata'],
                'total_execution_time': time.time() - start_time,
                'post_processed': processed_output != result['text']
            },
            'task_metadata': task_plan['metadata']
        }
        
        logger.info(f"Completed {task_type} task in {execution_result['metadata']['total_execution_time']:.2f}s")
        return execution_result
    
    def _get_max_tokens_for_task(self, task_type: str) -> int:
        """Get appropriate max tokens for task type."""
        token_limits = {
            'summary': 256,
            'glossary': 384,
            'questions': 512
        }
        return token_limits.get(task_type, 400)
    
    def _get_temperature_for_task(self, task_type: str) -> float:
        """Get appropriate temperature for task type."""
        temperatures = {
            'summary': 0.3,  # More deterministic for summaries
            'glossary': 0.2,  # Very deterministic for definitions
            'questions': 0.7   # More creative for questions
        }
        return temperatures.get(task_type, 0.5)
    
    def _post_process_output(self, task_type: str, output: str) -> str:
        """Post-process model output based on task type."""
        if task_type == 'summary':
            return self._clean_summary(output)
        elif task_type == 'glossary':
            return self._clean_glossary(output)
        elif task_type == 'questions':
            return self._clean_questions(output)
        else:
            return output.strip()
    
    def _clean_summary(self, output: str) -> str:
        """Clean and format summary output."""
        import re
        
        # Extract numbered points
        points = re.findall(r'^\d+\.\s*(.+)$', output, re.MULTILINE)
        
        if len(points) >= 5:
            # Take first 5 points
            formatted_points = []
            for i, point in enumerate(points[:5], 1):
                clean_point = point.strip()
                if not clean_point.endswith(('.', '!', '?')):
                    clean_point += '.'
                formatted_points.append(f"{i}. {clean_point}")
            return '\n'.join(formatted_points)
        else:
            # Return original if format is unexpected
            return output.strip()
    
    def _clean_glossary(self, output: str) -> str:
        """Clean and format glossary output."""
        import re
        
        # Extract term-definition pairs
        lines = output.strip().split('\n')
        formatted_terms = []
        
        for line in lines:
            line = line.strip()
            if ':' in line and len(line.split(':')) >= 2:
                term, definition = line.split(':', 1)
                term = term.strip()
                definition = definition.strip()
                if term and definition:
                    formatted_terms.append(f"{term}: {definition}")
        
        if formatted_terms:
            return '\n'.join(formatted_terms)
        else:
            return output.strip()
    
    def _clean_questions(self, output: str) -> str:
        """Clean and format questions output."""
        import re
        
        lines = output.strip().split('\n')
        formatted_questions = []
        
        question_num = 1
        current_question = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with Q1:, Q2:, etc.
            if re.match(r'^Q\d+:?\s*', line):
                # Save previous question
                if current_question:
                    formatted_questions.append('\n'.join(current_question))
                    current_question = []
                
                # Start new question
                clean_line = re.sub(r"^Q\d+:?\s*", "", line)
                current_question.append(f"Q{question_num}: {clean_line}")

                question_num += 1
            elif line and current_question:
                # Continuation of current question
                current_question.append(line)
        
        # Add last question
        if current_question:
            formatted_questions.append('\n'.join(current_question))
        
        if formatted_questions:
            return '\n\n'.join(formatted_questions)
        else:
            return output.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            'model_name': self.model_name,
            'device': self.device,
            'quantization': {
                '8bit': self.use_8bit,
                '4bit': self.use_4bit
            },
            'lora_adapter': self.lora_adapter_path is not None,
            'lora_adapter_path': self.lora_adapter_path
        }
        
        if self.model:
            try:
                # Get model size
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                info.update({
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'trainable_percentage': 100 * trainable_params / total_params if total_params > 0 else 0
                })
            except:
                logger.warning("Could not compute model parameter counts")
        
        return info


def create_executor(
    model_name: str = None,
    device: str = 'auto',
    use_lora: bool = True,
    lora_path: str = "models/lora-adapter"
) -> ModelExecutor:
    """
    Factory function to create model executor with common configurations.
    
    Args:
        model_name: Model name (None for auto-selection)
        device: Target device
        use_lora: Whether to use LoRA adapter
        lora_path: Path to LoRA adapter
        
    Returns:
        Configured ModelExecutor
    """
    # Auto-select model based on device
    if model_name is None:
        if device == 'cpu' or not torch.cuda.is_available():
            model_name = ModelExecutor.DEFAULT_MODELS['cpu']
        else:
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory > 20:  # > 20GB
                model_name = ModelExecutor.DEFAULT_MODELS['gpu_large']
            else:
                model_name = ModelExecutor.DEFAULT_MODELS['gpu_small']
    
    logger.info(f"Auto-selected model: {model_name}")
    
    # Determine quantization
    use_4bit = device != 'cpu' and torch.cuda.is_available()
    
    # Setup LoRA path
    lora_adapter_path = lora_path if use_lora and Path(lora_path).exists() else None
    
    return ModelExecutor(
        model_name=model_name,
        device=device,
        use_4bit=use_4bit,
        lora_adapter_path=lora_adapter_path
    )


if __name__ == "__main__":
    # Test the executor
    import sys
    
    # Test with simple prompt
    executor = create_executor(device='cpu')  # CPU for testing
    
    print("Model Info:")
    info = executor.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nTesting generation...")
    
    result = executor.generate_text(
        system_prompt="You are a helpful assistant.",
        user_prompt="Explain machine learning in one sentence.",
        max_tokens=50,
        temperature=0.7
    )
    
    print(f"Generated: {result['text']}")
    print(f"Time: {result['metadata']['generation_time']:.2f}s")
    print(f"Tokens/s: {result['metadata']['tokens_per_second']:.1f}") 