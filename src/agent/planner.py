"""
Planning and prompt management module for research paper analysis.

This module contains prompt templates and planning logic for coordinating
different analysis tasks (summarization, glossary, questions).
"""

import logging
from typing import Dict, List, Any, Optional
from enum import Enum
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Available analysis tasks."""
    SUMMARY = "summary"
    GLOSSARY = "glossary"
    QUESTIONS = "questions"
    ALL = "all"


class PromptTemplate:
    """Manages prompt templates for different tasks."""
    
    SYSTEM_PROMPTS = {
        TaskType.SUMMARY: """You are an expert academic researcher specializing in paper analysis. Your task is to create concise, accurate summaries of research papers. Focus on key contributions, methodology, and findings. Be precise and avoid hallucination.""",
        
        TaskType.GLOSSARY: """You are an expert technical writer creating glossaries for academic papers. Your task is to identify and define key technical terms that readers need to understand the paper. Provide clear, concise definitions.""",
        
        TaskType.QUESTIONS: """You are an expert educator creating exam questions based on research papers. Your task is to generate questions that test understanding of key concepts, methodology, and findings. Include various question types."""
    }
    
    TASK_PROMPTS = {
        TaskType.SUMMARY: """
Based on the following research paper content, create a concise 5-point summary. Each point should be exactly one sentence that captures a key aspect of the paper.

Requirements:
- Return exactly 5 numbered points (1. 2. 3. 4. 5.)
- Each point must be one complete sentence
- Cover: main contribution, methodology, key findings, implications, limitations
- Be factual and avoid speculation
- Keep each point under 30 words

Paper content:
{content}

Summary:""",
        
        TaskType.GLOSSARY: """
Based on the following research paper content, extract and define key technical terms that readers need to understand the paper.

Requirements:
- Return terms as "Term: Definition" format
- Include 5-8 terms maximum
- Each definition should be 1-2 sentences
- Focus on domain-specific technical terms
- Avoid common words or basic concepts
- Definitions should be clear and precise

Paper content:
{content}

Glossary:""",
        
        TaskType.QUESTIONS: """
Based on the following research paper content, generate 5 exam-style questions that test understanding of the material.

Requirements:
- Return exactly 5 questions numbered Q1-Q5
- Include one multiple choice question (MCQ) with options (A)-(D)
- For MCQ, indicate correct answer at the end [Answer: X]
- Mix question types: short answer, explain, describe, compare
- Questions should test understanding, not just memorization
- Cover different aspects: methodology, findings, implications

Paper content:
{content}

Questions:"""
    }
    
    @classmethod
    def get_system_prompt(cls, task_type: TaskType) -> str:
        """Get system prompt for task type."""
        return cls.SYSTEM_PROMPTS.get(task_type, cls.SYSTEM_PROMPTS[TaskType.SUMMARY])
    
    @classmethod
    def get_task_prompt(cls, task_type: TaskType, content: str) -> str:
        """Get formatted task prompt."""
        template = cls.TASK_PROMPTS.get(task_type, cls.TASK_PROMPTS[TaskType.SUMMARY])
        return template.format(content=content)


class TaskPlanner:
    """Plans and coordinates analysis tasks."""
    
    def __init__(self, max_content_length: int = 4000):
        """
        Initialize task planner.
        
        Args:
            max_content_length: Maximum characters to include in prompts
        """
        self.max_content_length = max_content_length
        self.prompt_template = PromptTemplate()
    
    def plan_tasks(self, task_type: TaskType) -> List[TaskType]:
        """
        Plan which tasks to execute.
        
        Args:
            task_type: Requested task type
            
        Returns:
            List of tasks to execute
        """
        if task_type == TaskType.ALL:
            return [TaskType.SUMMARY, TaskType.GLOSSARY, TaskType.QUESTIONS]
        else:
            return [task_type]
    
    def prepare_content(self, chunks: List[Dict[str, Any]], task_type: TaskType) -> str:
        """
        Prepare content from chunks for the specific task.
        
        Args:
            chunks: List of text chunks
            task_type: Type of task
            
        Returns:
            Formatted content string
        """
        # Combine chunks intelligently based on task
        if task_type == TaskType.SUMMARY:
            # For summary, prefer diverse chunks
            content = self._combine_chunks_diverse(chunks)
        elif task_type == TaskType.GLOSSARY:
            # For glossary, prefer technical content
            content = self._combine_chunks_technical(chunks)
        elif task_type == TaskType.QUESTIONS:
            # For questions, prefer methodological content
            content = self._combine_chunks_methodological(chunks)
        else:
            content = self._combine_chunks_default(chunks)
        
        # Truncate if too long
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "\n[Content truncated...]"
        
        return content
    
    def _combine_chunks_diverse(self, chunks: List[Dict[str, Any]]) -> str:
        """Combine chunks for diverse summary content."""
        combined = []
        total_length = 0
        
        # Sort by position to maintain flow
        sorted_chunks = sorted(chunks, key=lambda x: x.get('chunk_id', 0))
        
        for chunk in sorted_chunks:
            chunk_text = chunk['text']
            if total_length + len(chunk_text) > self.max_content_length:
                break
            combined.append(chunk_text)
            total_length += len(chunk_text)
        
        return "\n\n".join(combined)
    
    def _combine_chunks_technical(self, chunks: List[Dict[str, Any]]) -> str:
        """Combine chunks prioritizing technical content."""
        # Score chunks by technical term density
        technical_patterns = [
            r'\b(?:algorithm|model|method|approach|technique|framework)\b',
            r'\b(?:neural|network|learning|training|optimization)\b',
            r'\b(?:accuracy|performance|evaluation|metric|score)\b',
            r'\b(?:dataset|data|feature|parameter|hyperparameter)\b'
        ]
        
        import re
        
        scored_chunks = []
        for chunk in chunks:
            text = chunk['text'].lower()
            score = sum(len(re.findall(pattern, text)) for pattern in technical_patterns)
            scored_chunks.append((score, chunk))
        
        # Sort by technical score, then by length
        scored_chunks.sort(key=lambda x: (x[0], x[1].get('char_count', 0)), reverse=True)
        
        combined = []
        total_length = 0
        
        for score, chunk in scored_chunks:
            chunk_text = chunk['text']
            if total_length + len(chunk_text) > self.max_content_length:
                break
            combined.append(chunk_text)
            total_length += len(chunk_text)
        
        return "\n\n".join(combined)
    
    def _combine_chunks_methodological(self, chunks: List[Dict[str, Any]]) -> str:
        """Combine chunks prioritizing methodological content."""
        # Score chunks by methodological keywords
        method_patterns = [
            r'\b(?:experiment|study|analysis|investigation|evaluation)\b',
            r'\b(?:compare|comparison|baseline|benchmark|ablation)\b',
            r'\b(?:result|finding|conclusion|implication|limitation)\b',
            r'\b(?:method|methodology|approach|procedure|process)\b'
        ]
        
        import re
        
        scored_chunks = []
        for chunk in chunks:
            text = chunk['text'].lower()
            score = sum(len(re.findall(pattern, text)) for pattern in method_patterns)
            scored_chunks.append((score, chunk))
        
        scored_chunks.sort(key=lambda x: (x[0], x[1].get('char_count', 0)), reverse=True)
        
        combined = []
        total_length = 0
        
        for score, chunk in scored_chunks:
            chunk_text = chunk['text']
            if total_length + len(chunk_text) > self.max_content_length:
                break
            combined.append(chunk_text)
            total_length += len(chunk_text)
        
        return "\n\n".join(combined)
    
    def _combine_chunks_default(self, chunks: List[Dict[str, Any]]) -> str:
        """Default chunk combination strategy."""
        return self._combine_chunks_diverse(chunks)
    
    def create_prompt(self, task_type: TaskType, content: str) -> Dict[str, str]:
        """
        Create complete prompt for task.
        
        Args:
            task_type: Type of task
            content: Prepared content
            
        Returns:
            Dict with system and user prompts
        """
        system_prompt = self.prompt_template.get_system_prompt(task_type)
        user_prompt = self.prompt_template.get_task_prompt(task_type, content)
        
        return {
            'system': system_prompt,
            'user': user_prompt,
            'task_type': task_type.value,
            'content_length': len(content)
        }
    
    def validate_output(self, task_type: TaskType, output: str) -> Dict[str, Any]:
        """
        Validate model output against task requirements.
        
        Args:
            task_type: Type of task
            output: Model output
            
        Returns:
            Validation results
        """
        validation = {
            'valid': True,
            'issues': [],
            'score': 1.0
        }
        
        if task_type == TaskType.SUMMARY:
            validation.update(self._validate_summary(output))
        elif task_type == TaskType.GLOSSARY:
            validation.update(self._validate_glossary(output))
        elif task_type == TaskType.QUESTIONS:
            validation.update(self._validate_questions(output))
        
        return validation
    
    def _validate_summary(self, output: str) -> Dict[str, Any]:
        """Validate summary output."""
        import re
        
        issues = []
        score = 1.0
        
        # Check for 5 numbered points
        points = re.findall(r'^\d+\.\s*(.+)$', output, re.MULTILINE)
        if len(points) != 5:
            issues.append(f"Expected 5 points, found {len(points)}")
            score *= 0.8
        
        # Check point length
        for i, point in enumerate(points):
            words = len(point.split())
            if words > 35:
                issues.append(f"Point {i+1} too long ({words} words)")
                score *= 0.9
        
        # Check for proper sentences
        for i, point in enumerate(points):
            if not point.strip().endswith(('.', '!', '?')):
                issues.append(f"Point {i+1} not a complete sentence")
                score *= 0.9
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'score': score,
            'point_count': len(points)
        }
    
    def _validate_glossary(self, output: str) -> Dict[str, Any]:
        """Validate glossary output."""
        import re
        
        issues = []
        score = 1.0
        
        # Check for term: definition format
        terms = re.findall(r'^([^:]+):\s*(.+)$', output, re.MULTILINE)
        
        if len(terms) < 3:
            issues.append(f"Too few terms ({len(terms)}, expected 3-8)")
            score *= 0.7
        elif len(terms) > 10:
            issues.append(f"Too many terms ({len(terms)}, expected 3-8)")
            score *= 0.9
        
        # Check definition quality
        for term, definition in terms:
            if len(definition.split()) < 3:
                issues.append(f"Definition for '{term.strip()}' too short")
                score *= 0.9
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'score': score,
            'term_count': len(terms)
        }
    
    def _validate_questions(self, output: str) -> Dict[str, Any]:
        """Validate questions output."""
        import re
        
        issues = []
        score = 1.0
        
        # Check for 5 questions
        questions = re.findall(r'^Q\d+:?\s*(.+)$', output, re.MULTILINE)
        if len(questions) != 5:
            issues.append(f"Expected 5 questions, found {len(questions)}")
            score *= 0.8
        
        # Check for MCQ
        mcq_pattern = r'\(A\)|\(B\)|\(C\)|\(D\)'
        has_mcq = bool(re.search(mcq_pattern, output))
        if not has_mcq:
            issues.append("No multiple choice question found")
            score *= 0.9
        
        # Check for answer indication
        has_answer = bool(re.search(r'\[Answer:\s*[ABCD]\]', output))
        if has_mcq and not has_answer:
            issues.append("MCQ missing answer indication")
            score *= 0.9
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'score': score,
            'question_count': len(questions),
            'has_mcq': has_mcq,
            'has_answer': has_answer
        }


def plan_analysis(paper_data: Dict[str, Any], task_type: str) -> List[Dict[str, Any]]:
    """
    Plan complete analysis for paper.
    
    Args:
        paper_data: Processed paper data
        task_type: Type of analysis ('summary', 'glossary', 'questions', 'all')
        
    Returns:
        List of planned tasks with prompts
    """
    planner = TaskPlanner()
    
    # Convert string to enum
    try:
        task_enum = TaskType(task_type.lower())
    except ValueError:
        logger.warning(f"Unknown task type: {task_type}, defaulting to 'all'")
        task_enum = TaskType.ALL
    
    # Plan tasks
    planned_tasks = planner.plan_tasks(task_enum)
    
    # Prepare prompts for each task
    task_plans = []
    
    for task in planned_tasks:
        # Prepare content for this specific task
        content = planner.prepare_content(paper_data['text']['selected_chunks'], task)
        
        # Create prompt
        prompt_data = planner.create_prompt(task, content)
        
        task_plan = {
            'task_type': task.value,
            'prompt': prompt_data,
            'metadata': {
                'paper_title': paper_data['metadata'].get('title', 'Unknown'),
                'chunk_count': len(paper_data['text']['selected_chunks']),
                'content_length': len(content)
            }
        }
        
        task_plans.append(task_plan)
        
        logger.info(f"Planned {task.value} task with {len(content)} characters")
    
    return task_plans


if __name__ == "__main__":
    # Test the planner
    import json
    from pathlib import Path
    
    # Load sample processed paper data
    sample_data = {
        'metadata': {'title': 'Sample Paper on Machine Learning'},
        'text': {
            'selected_chunks': [
                {
                    'chunk_id': 0,
                    'text': 'This paper presents a novel approach to neural network training using reinforcement learning. The method achieves state-of-the-art performance on benchmark datasets.',
                    'char_count': 150
                },
                {
                    'chunk_id': 1,
                    'text': 'We introduce a new architecture called Adaptive Neural Networks that dynamically adjusts its structure during training. The key innovation is the use of meta-learning.',
                    'char_count': 160
                }
            ]
        }
    }
    
    # Test planning
    plans = plan_analysis(sample_data, 'all')
    
    print(f"Generated {len(plans)} task plans:")
    for plan in plans:
        print(f"- {plan['task_type']}: {plan['metadata']['content_length']} chars")
    
    # Save example plans
    output_path = "logs/example_plans.json"
    Path(output_path).parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(plans, f, indent=2)
    
    print(f"Example plans saved to: {output_path}")