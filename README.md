# Malnutrition Medical Assistant Chatbot

A domain-specific AI chatbot specialized in severe malnutrition treatment based on WHO guidelines, built using fine-tuned T5 Transformer model with a modern Streamlit interface.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Performance Metrics](#performance-metrics)
- [Project Structure](#project-structure)
- [Example Conversations](#example-conversations)
- [Demo Video](#demo-video)
- [Deployment](#deployment)
## **Video Link**: [Demo Video](https://drive.google.com/drive/folders/1HG0TTkYR40nizx1KDp0Cd6Ae0vI1uJF2?usp=sharing)]
## **Model Link**: [Model](https://drive.google.com/drive/folders/16ACpww0hKYLoweHjWArIs-t3wWqRKPCa?usp=drive_link)]

## 🎯 Overview

This project implements an intelligent chatbot that provides medical guidance on severe malnutrition treatment following WHO (World Health Organization) guidelines. The chatbot uses a fine-tuned T5 transformer model and includes robust domain filtering to ensure safe, relevant responses.

### Key Objectives
- Provide accurate medical information on malnutrition treatment
- Implement domain-specific safety boundaries
- Deliver user-friendly conversational interface
- Ensure response quality through NLP evaluation metrics

## 📊 Dataset

### Source
Custom malnutrition Q&A dataset (`malnutrition_qa.jsonl`) containing 52 conversational pairs.

### Dataset Characteristics
- **Format**: JSONL (JSON Lines)
- **Total Samples**: 48 question-answer pairs
- **Domain**: Severe malnutrition treatment protocols
- **Source**: WHO Malnutrition Guidelines

### Topics Covered
- Severe malnutrition classification and criteria
- Treatment protocols (F-75, F-100 diets)
- Emergency management (hypoglycemia, dehydration)
- Therapeutic feeding procedures
- Rehabilitation protocols
- Discharge criteria and follow-up
- Complications management

### Data Split
- **Training Set**: 80% (38 samples)
- **Test Set**: 20% (10 samples)

## 🏗️ Model Architecture

### Base Model
- **Model**: T5-small (Text-to-Text Transfer Transformer)
- **Parameters**: 60 million
- **Framework**: Hugging Face Transformers
- **Backend**: PyTorch

### Fine-tuning Configuration
- **Epochs**: 3
- **Batch Size**: 2
- **Learning Rate**: 5e-5
- **Max Input Length**: 512 tokens
- **Max Output Length**: 300 tokens
- **Optimizer**: AdamW
- **Weight Decay**: 0.01

### Generation Parameters
- **Num Beams**: 6
- **Repetition Penalty**: 3.0
- **Length Penalty**: 1.5
- **No Repeat N-gram Size**: 3

## ✨ Features

### Core Functionality
- Domain-specific responses specialized in malnutrition treatment
- Safety filtering blocks out-of-domain queries
- Real-time chat interface with modern UI
- Conversation history with timestamps
- Response time display
- Sample questions for quick testing

### Safety Features
- Domain boundary enforcement
- Dangerous topic blocking
- Non-medical query rejection
- Clear scope communication

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- 4GB+ RAM

### Steps
```bash
# Clone repository
git clone https://github.com/yourusername/malnutrition-chatbot.git
cd malnutrition-chatbot

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

## 💻 Usage

### Training the Model
```bash
# Quick training
python malnutrition_chatbot.py

# Hyperparameter tuning
python hyperparameter_tuning.py

# Comprehensive evaluation
python evaluation.py
```

### Running the Chatbot
```bash
# Streamlit web interface (Recommended)
streamlit run app.py

# Command line demo
python demo.py
```

Access at: `http://localhost:8501`

## 📈 Performance Metrics

### Evaluation Results

| Metric | Score | Description |
|--------|-------|-------------|
| BLEU Score | 0.35-0.45 | N-gram overlap with reference |
| ROUGE-L | 0.40-0.50 | Longest common subsequence |
| Domain Accuracy | 95%+ | In-domain acceptance rate |
| Safety Accuracy | 100% | Out-of-domain rejection rate |

### Hyperparameter Tuning

| Config | Epochs | Batch Size | BLEU |
|--------|--------|------------|------|
| 1 | 2 | 2 | 0.38 |
| 2 | 3 | 2 | 0.42 |
| 3 | 4 | 2 | 0.41 |
| 4 | 3 | 4 | 0.40 |

**Best**: 3 epochs, batch size 2

## 📁 Project Structure

```
Summative_ML_Technique1/
├── app.py                          # Streamlit interface
├── Notebook         # Core implementation                   
├── malnutrition_qa.jsonl           # Dataset
├── requirements.txt                # Dependencies
└── README.md                       # Documentation
```

## 💬 Example Conversations

### Example 1: In-Domain (Accepted)
```
User: What is severe malnutrition?
Assistant: (1.2s) Severe malnutrition includes children with weight-for-height 
below -3 SD or less than 70% of median reference values, or those with 
symmetrical oedema involving at least the feet.
```

### Example 2: Treatment Protocol
```
User: What is the F-75 diet?
Assistant: (1.5s) F-75 (75 kcal/100ml) is used during initial treatment phase. 
It contains essential minerals and vitamins for severely malnourished children.
```

### Example 3: Out-of-Domain (Blocked)
```
User: How do I change a car tire?
Assistant: (0.3s) I'm a medical assistant specialized in severe malnutrition 
treatment. I can only answer questions about malnutrition, therapeutic diets, 
and related medical topics from WHO guidelines.
```

### Example 4: Dangerous Medical Topic (Blocked)
```
User: How to treat a heart attack?
Assistant: (0.3s) I cannot provide advice about cardiac emergency. This is 
outside my specialized domain of severe malnutrition treatment. Please consult 
a healthcare professional for medical emergencies.
```

## 🎥 Demo Video

A 5-10 minute demo video showcasing:
- Chatbot functionality and user interface
- In-domain query handling
- Out-of-domain query rejection
- Response time and accuracy
- Key features and insights

**Video Link**: [Demo Video](https://drive.google.com/drive/folders/1HG0TTkYR40nizx1KDp0Cd6Ae0vI1uJF2?usp=sharing)]




## 📝 Key Insights

### Model Performance
- T5-small provides good balance between accuracy and speed
- Fine-tuning on domain-specific data significantly improves relevance
- Beam search with repetition penalty reduces redundancy

### Safety Implementation
- Keyword-based filtering effectively blocks out-of-domain queries
- Multi-layer safety checks prevent dangerous medical advice
- Clear communication maintains user trust

### User Experience
- Modern gradient UI enhances engagement
- Real-time response time display builds transparency
- Sample questions improve discoverability

## 🤝 Contributors

- Chol Daniel Deng
- Course: Machine Learning Technique 1
- Institution: African Leadership University

## 📄 License

This project is for educational purposes.

## 🔗 Links

- GitHub Repository: [Insert link]
- Demo Video: [Insert link]
- Dataset Source: WHO Malnutrition Guidelines

---

**Note**: This chatbot is for educational purposes only and should not replace professional medical advice.
