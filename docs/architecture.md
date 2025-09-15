# Research Paper Summarizer - Architecture Documentation

## System Overview

The Research Paper Summarizer is a comprehensive AI-powered tool that analyzes academic papers and generates summaries, glossaries, and exam questions using fine-tuned language models with LoRA (Low-Rank Adaptation).

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              User Interface Layer                               │
├─────────────────────────┬───────────────────────────┬───────────────────────────┤
│    CLI Interface        │    Streamlit Web App      │    REST API (Future)     │
│   (run_agent.py)        │   (streamlit_app.py)      │                           │
└─────────────────────────┴───────────────────────────┴───────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Agent Orchestration Layer                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                            Agent Controller                                     │
│  • Coordinates pipeline execution                                              │
│  • Manages task planning and execution                                         │
│  • Handles error recovery and logging                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                ┌────────────────────────┼────────────────────────┐
                ▼                        ▼                        ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│  Input Processor    │ │     Planner         │ │     Executor        │
│                     │ │                     │ │                     │
│ • PDF Extraction    │ │ • Task Planning     │ │ • Model Loading     │
│ • Text Cleaning     │ │ • Prompt Templates  │ │ • LoRA Integration  │
│ • Chunking Strategy │ │ • Content Selection │ │ • Text Generation   │
│ • Metadata Extract  │ │ • Validation Rules  │ │ • Post-processing   │
└─────────────────────┘ └─────────────────────┘ └─────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Model & Training Layer                             │
├─────────────────────────┬───────────────────────────┬───────────────────────────┤
│    Base Model           │    LoRA Adapter          │    Training Pipeline     │
│  • GPT-2/Phi-2/LLaMA   │  • Task-specific layers  │  • Dataset preparation   │
│  • Quantization         │  • Parameter efficient   │  • PEFT training         │
│  • Device management    │  • Fast adaptation       │  • Evaluation metrics    │
└─────────────────────────┴───────────────────────────┴───────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Storage & Evaluation Layer                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│  • Results storage (JSON/logs)                                                 │
│  • Model checkpoints                                                           │
│  • Evaluation metrics (ROUGE, BERTScore)                                       │
│  • Performance monitoring                                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Component Interaction Flow

### 1. Input Processing Pipeline

```
PDF Upload → Text Extraction → Cleaning → Chunking → Selection
     │              │              │           │          │
     │         [PyMuPDF]      [Regex]    [Token-based] [Top-K]
     │              │              │           │          │
     └─────────────────────────────────────────────────────┘
```

### 2. Task Planning and Execution

```
User Request → Task Planning → Prompt Generation → Model Execution → Post-processing
     │              │               │                     │              │
   [CLI/UI]    [Planner]      [Templates]           [Executor]      [Validation]
     │              │               │                     │              │
     └─────────────────────────────────────────────────────────────────────┘
```

### 3. Model Training Pipeline

```
Raw Data → Preprocessing → Training → Validation → Adapter Saving
    │           │            │           │             │
[JSONL]   [Dataset Prep] [LoRA+PEFT] [Metrics]   [Checkpoints]
    │           │            │           │             │
    └──────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Modular Architecture

**Why**: Separation of concerns enables independent development, testing, and maintenance of each component.

**Benefits**:
- Easy to extend with new tasks
- Simplified testing and debugging
- Clear responsibility boundaries
- Reusable components

### 2. LoRA for Fine-tuning

**Why**: Parameter-efficient fine-tuning reduces computational requirements while maintaining performance.

**Benefits**:
- Faster training (hours vs days)
- Lower memory requirements
- Easy adapter switching
- Preserves base model capabilities

**Implementation**:
```python
lora_config = LoraConfig(
    r=8,                    # Rank (controls adapter size)
    lora_alpha=32,          # Scaling parameter
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,      # Regularization
    task_type=TaskType.CAUSAL_LM
)
```

### 3. Multi-Model Support

**Why**: Different models offer different trade-offs between performance and computational requirements.

**Model Tiers**:
- **CPU-friendly**: GPT-2 (117M parameters)
- **GPU-moderate**: Phi-2 (2.7B parameters)
- **GPU-large**: LLaMA-2 7B (7B parameters)

### 4. Chunking Strategy

**Why**: Research papers are too long for model context windows, requiring intelligent text selection.

**Approaches**:
- **Token-based**: Fixed token count with overlap
- **Sentence-based**: Semantic boundaries
- **Mixed selection**: Combine early content + high-value chunks

### 5. Task-Specific Prompts

**Why**: Different tasks require different instruction formats and constraints.

**Templates**:
```python
SUMMARY_PROMPT = """
Based on the following research paper content, create a concise 5-point summary.
Requirements:
- Return exactly 5 numbered points
- Each point must be one complete sentence
- Cover: contribution, methodology, findings, implications, limitations
"""

GLOSSARY_PROMPT = """
Extract and define key technical terms from this paper.
Requirements:  
- Return as "Term: Definition" format
- Include 5-8 terms maximum
- Focus on domain-specific terminology
"""
```

## Technology Stack

### Core Dependencies

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| PDF Processing | PyMuPDF | 1.23+ | Text extraction |
| ML Framework | Transformers | 4.35+ | Model loading/inference |
| Fine-tuning | PEFT | 0.7+ | LoRA implementation |
| Training | Accelerate | 0.24+ | Distributed training |
| Evaluation | Rouge-Score | 0.1+ | Summary evaluation |
| Evaluation | Bert-Score | 0.3+ | Semantic similarity |
| UI Framework | Streamlit | 1.28+ | Web interface |
| Quantization | BitsAndBytes | Latest | Memory optimization |

### System Requirements

**Minimum (CPU)**:
- 8GB RAM
- 4 CPU cores
- 5GB disk space

**Recommended (GPU)**:
- 16GB GPU memory (RTX 4080/A100)
- 32GB system RAM
- 8+ CPU cores
- 50GB disk space

## Performance Characteristics

### Processing Speed (Approximate)

| Model | Device | Summary | Glossary | Questions | Total |
|-------|--------|---------|----------|-----------|-------|
| GPT-2 | CPU | 30s | 25s | 45s | 100s |
| Phi-2 | GPU | 8s | 6s | 12s | 26s |
| LLaMA-2 | GPU | 15s | 12s | 20s | 47s |

### Memory Usage

| Model | Base Memory | With LoRA | 4-bit Quantized |
|-------|-------------|-----------|-----------------|
| GPT-2 | 500MB | 520MB | N/A |
| Phi-2 | 5.5GB | 5.8GB | 3.2GB |
| LLaMA-2 | 13GB | 13.5GB | 7.5GB |

## Evaluation Framework

### Automatic Metrics

**Summary Quality**:
- ROUGE-1/2/L scores
- BERTScore F1
- Structure compliance
- Point count accuracy

**Glossary Quality**:
- Term coverage
- Definition quality
- Format compliance

**Question Quality**:
- Question diversity
- MCQ format validation
- Answer indication

### Human Evaluation Rubric

| Criterion | Weight | Scale | Description |
|-----------|--------|-------|-------------|
| Factual Accuracy | 30% | 1-5 | Information correctness |
| Completeness | 25% | 1-5 | Coverage of key points |
| Clarity | 20% | 1-5 | Readability and coherence |
| Format Compliance | 15% | 1-5 | Adherence to requirements |
| Usefulness | 10% | 1-5 | Practical value for users |

## Extensibility and Future Work

### Planned Extensions

1. **Multi-language Support**: Extend to non-English papers
2. **Advanced RAG**: Vector database integration (Chroma/FAISS)
3. **Domain Adaptation**: Field-specific models (CS, Biology, Physics)
4. **Collaborative Features**: Multi-user annotation and review
5. **API Service**: REST/GraphQL API for integration

### Plugin Architecture

```python
class TaskPlugin:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def process(self, content: str, config: dict) -> dict:
        """Process content and return results"""
        raise NotImplementedError
    
    def validate(self, output: str) -> dict:
        """Validate output quality"""
        raise NotImplementedError
```

## Deployment Considerations

### Local Development
- Use CPU-friendly models (GPT-2)
- Enable detailed logging
- Use small test datasets

### Production Deployment
- GPU instances with adequate memory
- Model quantization for efficiency
- Monitoring and alerting
- Backup and recovery procedures

### Scalability
- Batch processing capabilities
- Queue management for concurrent requests
- Model caching and warm starts
- Database optimization for large-scale usage

## Security and Ethics

### Data Privacy
- Local processing (no external API calls)
- Temporary file cleanup
- User consent for data storage

### Model Bias
- Diverse training data
- Regular bias evaluation
- Clear limitation documentation

### Responsible AI
- Hallucination warnings
- Human-in-the-loop validation
- Transparent model capabilities
- Proper attribution and licensing

## Maintenance and Monitoring

### Health Checks
- Model loading verification
- Memory usage monitoring
- Processing time tracking
- Error rate analysis

### Updates and Versioning
- Semantic versioning for releases
- Backward compatibility maintenance
- Migration scripts for data/models
- Documentation updates

This architecture provides a solid foundation for research paper analysis while remaining extensible and maintainable for future enhancements.