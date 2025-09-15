"""
Evaluation module for research paper analysis models.

This module implements evaluation metrics including ROUGE, BERTScore,
and custom metrics for assessing model performance on summarization,
glossary generation, and question generation tasks.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import re
import numpy as np
from dataclasses import dataclass
import argparse
import time

# Evaluation metrics
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logging.warning("rouge-score not installed, ROUGE metrics unavailable")

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    logging.warning("bert-score not installed, BERTScore unavailable")

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    task_type: str
    metric_name: str
    score: float
    details: Dict[str, Any]
    reference_count: int
    prediction_count: int


class BaseEvaluator:
    """Base class for task-specific evaluators."""
    
    def __init__(self, task_type: str):
        self.task_type = task_type
    
    def evaluate(self, predictions: List[str], references: List[str]) -> List[EvaluationResult]:
        """
        Evaluate predictions against references.
        
        Args:
            predictions: Model predictions
            references: Ground truth references
            
        Returns:
            List of evaluation results
        """
        raise NotImplementedError
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for evaluation."""
        return text.strip()


class SummaryEvaluator(BaseEvaluator):
    """Evaluator for summary generation task."""
    
    def __init__(self):
        super().__init__("summary")
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def evaluate(self, predictions: List[str], references: List[str]) -> List[EvaluationResult]:
        results = []
        
        # ROUGE scores
        if ROUGE_AVAILABLE:
            rouge_results = self._evaluate_rouge(predictions, references)
            results.extend(rouge_results)
        
        # BERTScore
        if BERTSCORE_AVAILABLE:
            bert_results = self._evaluate_bertscore(predictions, references)
            results.extend(bert_results)
        
        # Custom summary metrics
        custom_results = self._evaluate_summary_quality(predictions, references)
        results.extend(custom_results)
        
        return results
    
    def _evaluate_rouge(self, predictions: List[str], references: List[str]) -> List[EvaluationResult]:
        """Evaluate using ROUGE metrics."""
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            pred_clean = self.preprocess_text(pred)
            ref_clean = self.preprocess_text(ref)
            
            scores = self.rouge_scorer.score(ref_clean, pred_clean)
            
            for metric in rouge_scores:
                rouge_scores[metric].append(scores[metric].fmeasure)
        
        results = []
        for metric, scores in rouge_scores.items():
            if scores:
                results.append(EvaluationResult(
                    task_type=self.task_type,
                    metric_name=f"ROUGE-{metric.upper()}",
                    score=np.mean(scores),
                    details={
                        'scores': scores,
                        'std': np.std(scores),
                        'min': np.min(scores),
                        'max': np.max(scores)
                    },
                    reference_count=len(references),
                    prediction_count=len(predictions)
                ))
        
        return results
    
    def _evaluate_bertscore(self, predictions: List[str], references: List[str]) -> List[EvaluationResult]:
        """Evaluate using BERTScore."""
        try:
            P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)
            
            return [
                EvaluationResult(
                    task_type=self.task_type,
                    metric_name="BERTScore-F1",
                    score=F1.mean().item(),
                    details={
                        'precision': P.mean().item(),
                        'recall': R.mean().item(),
                        'f1_scores': F1.tolist(),
                        'std': F1.std().item()
                    },
                    reference_count=len(references),
                    prediction_count=len(predictions)
                )
            ]
        except Exception as e:
            logger.warning(f"BERTScore evaluation failed: {e}")
            return []
    
    def _evaluate_summary_quality(self, predictions: List[str], references: List[str]) -> List[EvaluationResult]:
        """Evaluate summary-specific quality metrics."""
        point_counts = []
        structure_scores = []
        
        for pred in predictions:
            # Count numbered points
            points = re.findall(r'^\d+\.\s*(.+)$', pred, re.MULTILINE)
            point_counts.append(len(points))
            
            # Structure score (0-1 based on expected format)
            structure_score = 0
            if len(points) == 5:  # Expected 5 points
                structure_score += 0.5
            
            # Check if points are complete sentences
            complete_sentences = sum(1 for point in points if point.strip().endswith(('.', '!', '?')))
            if points and complete_sentences == len(points):
                structure_score += 0.3
            
            # Check point length (reasonable but not too long)
            reasonable_length = sum(1 for point in points if 5 <= len(point.split()) <= 35)
            if points and reasonable_length == len(points):
                structure_score += 0.2
            
            structure_scores.append(structure_score)
        
        return [
            EvaluationResult(
                task_type=self.task_type,
                metric_name="Point Count Accuracy",
                score=np.mean([1.0 if count == 5 else 0.0 for count in point_counts]),
                details={
                    'point_counts': point_counts,
                    'expected_points': 5,
                    'avg_points': np.mean(point_counts)
                },
                reference_count=len(references),
                prediction_count=len(predictions)
            ),
            EvaluationResult(
                task_type=self.task_type,
                metric_name="Structure Quality",
                score=np.mean(structure_scores),
                details={
                    'structure_scores': structure_scores,
                    'criteria': ['5_points', 'complete_sentences', 'reasonable_length']
                },
                reference_count=len(references),
                prediction_count=len(predictions)
            )
        ]


class GlossaryEvaluator(BaseEvaluator):
    """Evaluator for glossary generation task."""
    
    def __init__(self):
        super().__init__("glossary")
    
    def evaluate(self, predictions: List[str], references: List[str]) -> List[EvaluationResult]:
        results = []
        
        # Coverage metrics
        coverage_results = self._evaluate_coverage(predictions, references)
        results.extend(coverage_results)
        
        # Format quality
        format_results = self._evaluate_format_quality(predictions)
        results.extend(format_results)
        
        # Term extraction accuracy
        if len(predictions) == len(references):
            term_results = self._evaluate_term_accuracy(predictions, references)
            results.extend(term_results)
        
        return results
    
    def _evaluate_coverage(self, predictions: List[str], references: List[str]) -> List[EvaluationResult]:
        """Evaluate term coverage."""
        if len(predictions) != len(references):
            logger.warning("Prediction and reference counts don't match for coverage evaluation")
            return []
        
        coverage_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_terms = self._extract_terms(pred)
            ref_terms = self._extract_terms(ref)
            
            if ref_terms:
                # Calculate overlap
                common_terms = set(pred_terms) & set(ref_terms)
                coverage = len(common_terms) / len(ref_terms)
            else:
                coverage = 0.0
            
            coverage_scores.append(coverage)
        
        return [
            EvaluationResult(
                task_type=self.task_type,
                metric_name="Term Coverage",
                score=np.mean(coverage_scores),
                details={
                    'coverage_scores': coverage_scores,
                    'std': np.std(coverage_scores)
                },
                reference_count=len(references),
                prediction_count=len(predictions)
            )
        ]
    
    def _evaluate_format_quality(self, predictions: List[str]) -> List[EvaluationResult]:
        """Evaluate glossary format quality."""
        format_scores = []
        term_counts = []
        
        for pred in predictions:
            terms = self._extract_terms_with_definitions(pred)
            term_counts.append(len(terms))
            
            format_score = 0
            
            # Check if we have reasonable number of terms (3-8)
            if 3 <= len(terms) <= 8:
                format_score += 0.4
            
            # Check if all terms have definitions
            valid_definitions = sum(1 for _, defn in terms if len(defn.strip()) > 10)
            if terms and valid_definitions == len(terms):
                format_score += 0.4
            
            # Check format consistency (Term: Definition)
            proper_format = re.findall(r'^[^:]+:\s*.+$', pred, re.MULTILINE)
            if len(proper_format) >= len(terms) * 0.8:  # At least 80% proper format
                format_score += 0.2
            
            format_scores.append(format_score)
        
        return [
            EvaluationResult(
                task_type=self.task_type,
                metric_name="Format Quality",
                score=np.mean(format_scores),
                details={
                    'format_scores': format_scores,
                    'term_counts': term_counts,
                    'avg_terms': np.mean(term_counts)
                },
                reference_count=0,  # Not applicable for format
                prediction_count=len(predictions)
            )
        ]
    
    def _evaluate_term_accuracy(self, predictions: List[str], references: List[str]) -> List[EvaluationResult]:
        """Evaluate term extraction accuracy."""
        precision_scores = []
        recall_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_terms = set(self._extract_terms(pred))
            ref_terms = set(self._extract_terms(ref))
            
            if pred_terms:
                precision = len(pred_terms & ref_terms) / len(pred_terms)
            else:
                precision = 0.0
            
            if ref_terms:
                recall = len(pred_terms & ref_terms) / len(ref_terms)
            else:
                recall = 0.0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        f1_scores = [
            2 * p * r / (p + r) if (p + r) > 0 else 0.0
            for p, r in zip(precision_scores, recall_scores)
        ]
        
        return [
            EvaluationResult(
                task_type=self.task_type,
                metric_name="Term Precision",
                score=np.mean(precision_scores),
                details={'precision_scores': precision_scores},
                reference_count=len(references),
                prediction_count=len(predictions)
            ),
            EvaluationResult(
                task_type=self.task_type,
                metric_name="Term Recall",
                score=np.mean(recall_scores),
                details={'recall_scores': recall_scores},
                reference_count=len(references),
                prediction_count=len(predictions)
            ),
            EvaluationResult(
                task_type=self.task_type,
                metric_name="Term F1-Score",
                score=np.mean(f1_scores),
                details={'f1_scores': f1_scores},
                reference_count=len(references),
                prediction_count=len(predictions)
            )
        ]
    
    def _extract_terms(self, text: str) -> List[str]:
        """Extract terms from glossary text."""
        terms = []
        for line in text.split('\n'):
            if ':' in line:
                term = line.split(':', 1)[0].strip().lower()
                if term:
                    terms.append(term)
        return terms
    
    def _extract_terms_with_definitions(self, text: str) -> List[Tuple[str, str]]:
        """Extract terms with their definitions."""
        terms = []
        for line in text.split('\n'):
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    term = parts[0].strip()
                    definition = parts[1].strip()
                    if term and definition:
                        terms.append((term, definition))
        return terms


class QuestionEvaluator(BaseEvaluator):
    """Evaluator for question generation task."""
    
    def __init__(self):
        super().__init__("questions")
    
    def evaluate(self, predictions: List[str], references: List[str]) -> List[EvaluationResult]:
        results = []
        
        # Question format quality
        format_results = self._evaluate_format_quality(predictions)
        results.extend(format_results)
        
        # Question diversity
        diversity_results = self._evaluate_diversity(predictions)
        results.extend(diversity_results)
        
        # MCQ format validation
        mcq_results = self._evaluate_mcq_format(predictions)
        results.extend(mcq_results)
        
        return results
    
    def _evaluate_format_quality(self, predictions: List[str]) -> List[EvaluationResult]:
        """Evaluate question format quality."""
        format_scores = []
        question_counts = []
        
        for pred in predictions:
            questions = self._extract_questions(pred)
            question_counts.append(len(questions))
            
            format_score = 0
            
            # Check if we have 5 questions
            if len(questions) == 5:
                format_score += 0.4
            
            # Check if questions are properly numbered
            numbered_questions = re.findall(r'^Q\d+:?\s*(.+)$', pred, re.MULTILINE)
            if len(numbered_questions) >= len(questions) * 0.8:
                format_score += 0.3
            
            # Check if questions end with question marks or are complete
            valid_questions = sum(1 for q in questions if len(q.strip()) > 10)
            if questions and valid_questions >= len(questions) * 0.8:
                format_score += 0.3
            
            format_scores.append(format_score)
        
        return [
            EvaluationResult(
                task_type=self.task_type,
                metric_name="Format Quality",
                score=np.mean(format_scores),
                details={
                    'format_scores': format_scores,
                    'question_counts': question_counts,
                    'expected_questions': 5
                },
                reference_count=0,
                prediction_count=len(predictions)
            )
        ]
    
    def _evaluate_diversity(self, predictions: List[str]) -> List[EvaluationResult]:
        """Evaluate question type diversity."""
        diversity_scores = []
        
        question_starters = [
            'what', 'how', 'why', 'when', 'where', 'which', 'who',
            'explain', 'describe', 'compare', 'analyze', 'discuss'
        ]
        
        for pred in predictions:
            questions = self._extract_questions(pred)
            
            if not questions:
                diversity_scores.append(0.0)
                continue
            
            # Count unique question starters
            starters_used = set()
            for question in questions:
                question_lower = question.lower().strip()
                for starter in question_starters:
                    if question_lower.startswith(starter):
                        starters_used.add(starter)
                        break
            
            # Diversity score based on unique starters
            diversity = len(starters_used) / min(len(questions), len(question_starters))
            diversity_scores.append(diversity)
        
        return [
            EvaluationResult(
                task_type=self.task_type,
                metric_name="Question Diversity",
                score=np.mean(diversity_scores),
                details={
                    'diversity_scores': diversity_scores,
                    'question_starters': question_starters
                },
                reference_count=0,
                prediction_count=len(predictions)
            )
        ]
    
    def _evaluate_mcq_format(self, predictions: List[str]) -> List[EvaluationResult]:
        """Evaluate MCQ format compliance."""
        mcq_scores = []
        
        for pred in predictions:
            mcq_score = 0
            
            # Check for MCQ options (A), (B), (C), (D)
            options = re.findall(r'\([ABCD]\)', pred)
            if len(options) >= 4:  # At least 4 options
                mcq_score += 0.5
            
            # Check for answer indication
            answer_match = re.search(r'\[Answer:\s*[ABCD]\]', pred)
            if answer_match:
                mcq_score += 0.5
            
            mcq_scores.append(mcq_score)
        
        # Check if at least one prediction has MCQ
        has_mcq = any(score > 0 for score in mcq_scores)
        mcq_presence_score = 1.0 if has_mcq else 0.0
        
        return [
            EvaluationResult(
                task_type=self.task_type,
                metric_name="MCQ Format Quality",
                score=np.mean(mcq_scores),
                details={
                    'mcq_scores': mcq_scores,
                    'criteria': ['4_options', 'answer_indicated']
                },
                reference_count=0,
                prediction_count=len(predictions)
            ),
            EvaluationResult(
                task_type=self.task_type,
                metric_name="MCQ Presence",
                score=mcq_presence_score,
                details={
                    'has_mcq': has_mcq,
                    'mcq_count': sum(1 for score in mcq_scores if score > 0)
                },
                reference_count=0,
                prediction_count=len(predictions)
            )
        ]
    
    def _extract_questions(self, text: str) -> List[str]:
        """Extract questions from text."""
        questions = []
        
        # Look for numbered questions
        question_matches = re.findall(r'^Q\d+:?\s*(.+)$', text, re.MULTILINE)
        questions.extend(question_matches)
        
        # If no numbered questions, look for question marks
        if not questions:
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line.endswith('?') and len(line) > 10:
                    questions.append(line)
        
        return questions


class ModelEvaluator:
    """Main evaluator that coordinates task-specific evaluators."""
    
    def __init__(self):
        self.evaluators = {
            'summary': SummaryEvaluator(),
            'glossary': GlossaryEvaluator(),
            'questions': QuestionEvaluator()
        }
    
    def evaluate_predictions(
        self,
        predictions: Dict[str, List[str]],
        references: Dict[str, List[str]] = None
    ) -> Dict[str, List[EvaluationResult]]:
        """
        Evaluate predictions for multiple tasks.
        
        Args:
            predictions: Dict mapping task types to predictions
            references: Dict mapping task types to references (optional)
            
        Returns:
            Dict mapping task types to evaluation results
        """
        all_results = {}
        
        for task_type, task_predictions in predictions.items():
            if task_type not in self.evaluators:
                logger.warning(f"No evaluator available for task: {task_type}")
                continue
            
            task_references = references.get(task_type, []) if references else []
            
            logger.info(f"Evaluating {task_type}: {len(task_predictions)} predictions")
            
            evaluator = self.evaluators[task_type]
            results = evaluator.evaluate(task_predictions, task_references)
            
            all_results[task_type] = results
        
        return all_results
    
    def load_predictions_from_results(self, results_path: str) -> Dict[str, List[str]]:
        """Load predictions from analysis results file."""
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        predictions = {}
        
        for task_result in data.get('task_results', []):
            task_type = task_result['task_type']
            output = task_result['output']
            
            if task_type not in predictions:
                predictions[task_type] = []
            
            predictions[task_type].append(output)
        
        return predictions
    
    def load_references_from_dataset(self, dataset_path: str) -> Dict[str, List[str]]:
        """Load references from test dataset."""
        references = {'summary': [], 'glossary': [], 'questions': []}
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    instruction = record.get('instruction', '').lower()
                    response = record.get('response', '')
                    
                    if 'summary' in instruction:
                        references['summary'].append(response)
                    elif 'glossary' in instruction or 'terms' in instruction:
                        references['glossary'].append(response)
                    elif 'question' in instruction:
                        references['questions'].append(response)
        
        return references
    
    def save_evaluation_results(self, results: Dict[str, List[EvaluationResult]], output_path: str):
        """Save evaluation results to file."""
        output_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'evaluation_results': {},
            'summary': {}
        }
        
        # Convert results to serializable format
        for task_type, task_results in results.items():
            output_data['evaluation_results'][task_type] = []
            
            for result in task_results:
                output_data['evaluation_results'][task_type].append({
                    'metric_name': result.metric_name,
                    'score': result.score,
                    'details': result.details,
                    'reference_count': result.reference_count,
                    'prediction_count': result.prediction_count
                })
        
        # Create summary
        for task_type, task_results in results.items():
            output_data['summary'][task_type] = {}
            for result in task_results:
                output_data['summary'][task_type][result.metric_name] = result.score
        
        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to: {output_path}")
    
    def print_results(self, results: Dict[str, List[EvaluationResult]]):
        """Print evaluation results in a formatted way."""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        for task_type, task_results in results.items():
            print(f"\n{task_type.upper()} TASK:")
            print("-" * 30)
            
            for result in task_results:
                print(f"{result.metric_name}: {result.score:.4f}")
                
                # Show additional details for key metrics
                if 'ROUGE' in result.metric_name or 'BERTScore' in result.metric_name:
                    if 'std' in result.details:
                        print(f"  (std: {result.details['std']:.4f})")
        
        print("\n" + "="*60)


def main():
    """CLI for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate research paper analysis model")
    
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        help='Path to predictions file (analysis results JSON)'
    )
    
    parser.add_argument(
        '--references', '-r',
        type=str,
        help='Path to references file (test dataset JSONL)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='logs/eval_results.json',
        help='Output path for evaluation results'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        choices=['summary', 'glossary', 'questions', 'all'],
        default='all',
        help='Task to evaluate (default: all)'
    )
    
    args = parser.parse_args()
    
    try:
        evaluator = ModelEvaluator()
        
        # Load predictions
        if args.predictions:
            predictions = evaluator.load_predictions_from_results(args.predictions)
        else:
            # Load from last output
            last_output_path = "logs/last_output.json"
            if Path(last_output_path).exists():
                predictions = evaluator.load_predictions_from_results(last_output_path)
            else:
                print("Error: No predictions file specified and logs/last_output.json not found")
                return
        
        # Load references if available
        references = None
        if args.references and Path(args.references).exists():
            references = evaluator.load_references_from_dataset(args.references)
        
        # Filter by task if specified
        if args.task != 'all':
            predictions = {k: v for k, v in predictions.items() if k == args.task}
            if references:
                references = {k: v for k, v in references.items() if k == args.task}
        
        # Evaluate
        results = evaluator.evaluate_predictions(predictions, references)
        
        # Print results
        evaluator.print_results(results)
        
        # Save results
        evaluator.save_evaluation_results(results, args.output)
        
        print(f"\nDetailed results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()