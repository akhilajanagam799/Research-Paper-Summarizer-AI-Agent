"""
Command-line interface for the Research Paper Summarizer agent.

This script provides a CLI for running the complete paper analysis pipeline
including PDF processing, task planning, and model execution.
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.input_processor import process_paper
from agent.planner import plan_analysis, TaskPlanner
from agent.executor import create_executor, ModelExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_results(results: Dict[str, Any], output_dir: str = "logs") -> str:
    """
    Save analysis results to JSON file.
    
    Args:
        results: Analysis results
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_results_{timestamp}.json"
    filepath = Path(output_dir) / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {filepath}")
    return str(filepath)


def print_results(results: Dict[str, Any]):
    """Print formatted analysis results."""
    print("\n" + "="*60)
    print("RESEARCH PAPER ANALYSIS RESULTS")
    print("="*60)
    
    # Paper info
    metadata = results.get('paper_metadata', {})
    print(f"\nPaper: {metadata.get('title', 'Unknown Title')}")
    print(f"Pages: {results.get('statistics', {}).get('total_pages', 'Unknown')}")
    
    # Task results
    for task_result in results.get('task_results', []):
        task_type = task_result['task_type'].upper()
        output = task_result['output']
        
        print(f"\n{'-'*40}")
        print(f"{task_type}")
        print(f"{'-'*40}")
        print(output)
        
        # Show timing info
        metadata = task_result.get('metadata', {})
        if 'total_execution_time' in metadata:
            print(f"\n[Generated in {metadata['total_execution_time']:.1f}s]")
    
    # Overall statistics
    print(f"\n{'-'*40}")
    print("EXECUTION SUMMARY")
    print(f"{'-'*40}")
    
    total_time = results.get('total_time', 0)
    print(f"Total analysis time: {total_time:.1f}s")
    
    model_info = results.get('model_info', {})
    print(f"Model: {model_info.get('model_name', 'Unknown')}")
    print(f"Device: {model_info.get('device', 'Unknown')}")
    if model_info.get('lora_adapter'):
        print("LoRA adapter: Enabled")


def run_analysis(
    input_path: str,
    task_type: str = 'all',
    model_name: str = None,
    device: str = 'auto',
    use_lora: bool = True,
    output_dir: str = "logs",
    chunk_method: str = 'token',
    selection_method: str = 'mixed',
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Run complete paper analysis pipeline.
    
    Args:
        input_path: Path to PDF file
        task_type: Analysis task ('summary', 'glossary', 'questions', 'all')
        model_name: Model to use (None for auto-selection)
        device: Device for inference
        use_lora: Whether to use LoRA adapter
        output_dir: Output directory for results
        chunk_method: Text chunking method
        selection_method: Chunk selection method
        top_k: Number of chunks to select
        
    Returns:
        Complete analysis results
    """
    start_time = time.time()
    
    try:
        # Step 1: Process PDF
        logger.info(f"Processing PDF: {input_path}")
        paper_data = process_paper(
            pdf_path=input_path,
            chunk_method=chunk_method,
            selection_method=selection_method,
            top_k=top_k
        )
        
        # Step 2: Plan analysis tasks
        logger.info(f"Planning {task_type} analysis")
        task_plans = plan_analysis(paper_data, task_type)
        
        # Step 3: Initialize model executor
        logger.info("Loading model...")
        executor = create_executor(
            model_name=model_name,
            device=device,
            use_lora=use_lora
        )
        
        # Step 4: Execute tasks
        task_results = []
        planner = TaskPlanner()
        
        for task_plan in task_plans:
            logger.info(f"Executing {task_plan['task_type']} task")
            
            # Execute task
            result = executor.execute_task(task_plan)
            
            # Validate output
            from agent.planner import TaskType
            validation = planner.validate_output(
                TaskType(task_plan['task_type']), 
                result['output']
            )
            result['validation'] = validation
            
            task_results.append(result)
            
            # Log validation results
            if not validation['valid']:
                logger.warning(f"Validation issues for {task_plan['task_type']}: {validation['issues']}")
        
        # Step 5: Compile results
        total_time = time.time() - start_time
        
        results = {
            'input_file': input_path,
            'analysis_type': task_type,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_time': total_time,
            'paper_metadata': paper_data['metadata'],
            'statistics': paper_data['statistics'],
            'task_results': task_results,
            'model_info': executor.get_model_info(),
            'processing_config': {
                'chunk_method': chunk_method,
                'selection_method': selection_method,
                'top_k': top_k,
                'use_lora': use_lora
            }
        }
        
        # Step 6: Save results
        output_path = save_results(results, output_dir)
        results['output_path'] = output_path
        
        # Also save as "last_output.json" for easy access
        latest_path = Path(output_dir) / "last_output.json"
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis completed in {total_time:.1f}s")
        return results
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Research Paper Summarizer - AI-powered paper analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_agent.py --input paper.pdf --task summary
  python run_agent.py --input paper.pdf --task all --model microsoft/phi-2
  python run_agent.py --input paper.pdf --task glossary --device cpu --no-lora
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help="Path to input PDF file"
    )
    
    # Optional arguments
    parser.add_argument(
        '--task', '-t',
        type=str,
        choices=['summary', 'glossary', 'questions', 'all'],
        default='all',
        help="Analysis task to perform (default: all)"
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help="Model name (default: auto-select based on device)"
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help="Device for inference (default: auto)"
    )
    
    parser.add_argument(
        '--no-lora',
        action='store_true',
        help="Disable LoRA adapter"
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='logs',
        help="Output directory for results (default: logs)"
    )
    
    parser.add_argument(
        '--chunk-method',
        type=str,
        choices=['token', 'sentence'],
        default='token',
        help="Text chunking method (default: token)"
    )
    
    parser.add_argument(
        '--selection-method',
        type=str,
        choices=['length', 'position', 'mixed'],
        default='mixed',
        help="Chunk selection method (default: mixed)"
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help="Number of top chunks to select (default: 5)"
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help="Reduce output verbosity"
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Increase output verbosity"
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if not args.input.lower().endswith('.pdf'):
        print(f"Warning: Input file may not be a PDF: {args.input}")
    
    try:
        # Run analysis
        results = run_analysis(
            input_path=args.input,
            task_type=args.task,
            model_name=args.model,
            device=args.device,
            use_lora=not args.no_lora,
            output_dir=args.output_dir,
            chunk_method=args.chunk_method,
            selection_method=args.selection_method,
            top_k=args.top_k
        )
        
        # Print results unless quiet
        if not args.quiet:
            print_results(results)
        
        print(f"\nResults saved to: {results['output_path']}")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()