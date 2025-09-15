# Demo Recording Instructions

## 🎥 Video Demo (1-2 minutes)

### Recording Setup
- **Resolution**: 1920x1080 minimum
- **Frame rate**: 30fps
- **Audio**: Clear narration with noise cancellation
- **Screen**: Close unnecessary applications

### Demo Script

**[0:00-0:15] Introduction**
```
"Hi! I'm demonstrating my Research Paper Summarizer that uses LoRA fine-tuning 
to analyze academic papers and generate summaries, glossaries, and exam questions."
```

**[0:15-0:30] Upload Paper**
- Open Streamlit interface
- Upload sample PDF
- Show PDF preview in left panel

```
"I'll upload this machine learning research paper. The interface shows a split-screen 
with PDF preview on the left and analysis results on the right."
```

**[0:30-0:50] Generate Analysis**
- Toggle "Use LoRA Adapter" ON
- Select model (show dropdown)
- Click "Analyze Paper"
- Show processing indicator

```
"I'm enabling my fine-tuned LoRA adapter and running the analysis. 
The model extracts key content and processes it through three specialized tasks."
```

**[0:50-1:30] Show Results**
- Navigate through Summary tab
- Switch to Glossary tab
- Review Questions tab
- Highlight one MCQ example

```
"Here's the 5-point summary capturing the main contributions. 
The glossary defines technical terms like 'transformer architecture'. 
And here are exam questions including this multiple choice about attention mechanisms."
```

**[1:30-1:45] Evaluation & Export**
- Show evaluation metrics
- Click "Download Report"
- Mention training process

```
"The system provides ROUGE and BERTScore evaluation metrics. 
I can export everything as a PDF report. The LoRA adapter was trained 
on academic paper data using PEFT for efficient fine-tuning."
```

### 📸 Required Screenshots

#### Screenshot 1: Main Interface
- **Filename**: `demo_main_interface.png`
- **Content**: Streamlit app with PDF uploaded, all tabs visible
- **Caption**: "Split-screen interface with PDF preview and analysis tabs"

#### Screenshot 2: Summary Output
- **Filename**: `demo_summary_output.png`
- **Content**: Summary tab showing 5 numbered points
- **Caption**: "AI-generated 5-point research paper summary"

#### Screenshot 3: Glossary
- **Filename**: `demo_glossary.png`
- **Content**: Glossary tab with term definitions
- **Caption**: "Technical term glossary with concise definitions"

#### Screenshot 4: Questions
- **Filename**: `demo_questions.png`
- **Content**: Questions tab showing mixed question types
- **Caption**: "Generated exam questions including multiple choice"

#### Screenshot 5: Evaluation Metrics
- **Filename**: `demo_evaluation.png`
- **Content**: Evaluation results or metrics dashboard
- **Caption**: "ROUGE and BERTScore evaluation metrics"

#### Screenshot 6: CLI Output
- **Filename**: `demo_cli_output.png`
- **Content**: Terminal showing CLI command results
- **Caption**: "Command-line interface for batch processing"

### 📋 Demo Checklist

**Before Recording:**
- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] Sample PDF downloaded
- [ ] Streamlit app launches without errors
- [ ] LoRA adapter trained (or using base model)
- [ ] Audio levels tested

**During Recording:**
- [ ] Smooth mouse movements
- [ ] Pause between major actions
- [ ] Clear narration without filler words
- [ ] Show loading states briefly
- [ ] Highlight key features

**After Recording:**
- [ ] Video under 2 minutes
- [ ] All screenshots captured
- [ ] Audio sync checked
- [ ] Export in MP4 format
- [ ] File size under 50MB

### 🎨 Visual Tips

- Use **bright cursor** for visibility
- **Zoom in** on important UI elements
- **Highlight** key outputs with cursor
- **Smooth scrolling** when showing results
- **Consistent pacing** between sections

### 📤 Submission Format

```
demo_video.mp4           # Main demo video
screenshots/             # Folder with all 6 screenshots
├── demo_main_interface.png
├── demo_summary_output.png
├── demo_glossary.png
├── demo_questions.png
├── demo_evaluation.png
└── demo_cli_output.png
```

### 📝 Video Description Template

```
Research Paper Summarizer - AI Analysis Tool

This tool uses LoRA fine-tuning to analyze academic papers and generate:
• 5-point summaries of key findings
• Technical term glossaries  
• Exam-style questions (including MCQ)

Features demonstrated:
- PDF upload and preview
- Real-time AI analysis
- Multi-tab results interface
- Evaluation metrics
- Report export functionality

Tech stack: Python, Transformers, PEFT, Streamlit, PyMuPDF
Evaluation: ROUGE scores, BERTScore metrics
```