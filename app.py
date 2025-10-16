# app.py
import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Page configuration
st.set_page_config(
    page_title="Malnutrition Medical Assistant",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .safety-alert {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .medical-response {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border: none;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 10px 30px rgba(240, 147, 251, 0.3);
        animation: slideIn 0.3s ease-out;
    }
    
    .blocked-response {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        border: none;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 10px 30px rgba(250, 112, 154, 0.3);
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .stTextArea>div>div>textarea {
        border-radius: 12px;
        border: 2px solid #e5e7eb;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea>div>div>textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .metric-card {
        background: white !important;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    h3 {
        color: #1f2937 !important;
    }
    
    div[data-testid="stExpander"] {
        background: white !important;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    div[data-testid="stExpander"] > div {
        background: white !important;
    }
    
    .stApp {
        background: linear-gradient(180deg, #f9fafb 0%, #ffffff 100%);
    }
</style>
""", unsafe_allow_html=True)

# Domain Safety Guard Class
class MedicalDomainGuard:
    def __init__(self):
        self.domain_keywords = [
            'malnutrition', 'undernutrition', 'f-75', 'f-100', 'resomal', 
            'hypoglycemia', 'dehydration', 'oedema', 'edema', 'kwashiorkor', 
            'marasmus', 'discharge', 'treatment', 'diet', 'feeding', 
            'transfusion', 'anemia', 'child', 'children', 'severely',
            'weight-for-height', 'muac', 'nutrition', 'therapeutic', 'rehabilitation',
            'who', 'guidelines', 'protocol', 'initial phase', 'rehabilitation phase'
        ]
        
        self.dangerous_out_of_scope = {
            'heart attack': 'cardiac emergency',
            'stroke': 'neurological emergency', 
            'cancer': 'oncological condition',
            'surgery': 'surgical procedure',
            'pregnancy': 'obstetric care',
            'diabetes': 'endocrine disorder',
            'malaria': 'infectious disease',
            'cpr': 'emergency procedure',
            'overdose': 'toxicology emergency',
            'prescription': 'medication management',
            'covid': 'infectious disease',
            'hiv': 'immunological condition',
            'heart failure': 'cardiovascular condition'
        }
        
        self.non_medical_indicators = [
            'car', 'tire', 'bake', 'cake', 'capital', 'france', 'math',
            'homework', 'name', 'created', 'economic', 'sport', 'game',
            'movie', 'music', 'weather', 'cooking', 'travel', 'shopping',
            'weather', 'sports', 'entertainment', 'politics', 'religion',
            'programming', 'code', 'computer', 'phone', 'technology'
        ]
    
    def is_in_domain(self, question):
        question_lower = question.lower()
        
        # Quick keyword check
        keyword_match = any(keyword in question_lower for keyword in self.domain_keywords)
        if keyword_match:
            return True, "in_domain"
        
        # Check for dangerous medical topics outside our scope
        for danger_keyword, danger_type in self.dangerous_out_of_scope.items():
            if danger_keyword in question_lower:
                return False, f"dangerous_medical:{danger_type}"
        
        # Check for non-medical topics
        for indicator in self.non_medical_indicators:
            if indicator in question_lower:
                return False, "non_medical"
        
        return False, "uncertain_medical"

# In your Streamlit app, update the model path
import os

def load_model():
    possible_paths = [
        "./malnutrition-t5-final",
        "malnutrition-t5-final",
        "../malnutrition-t5-final"
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                # Simply load directly to device - let transformers handle it
                model = T5ForConditionalGeneration.from_pretrained(
                    path,
                    device_map="auto" if torch.cuda.is_available() else None,
                    torch_dtype=torch.float32
                )
                tokenizer = T5Tokenizer.from_pretrained(path)
                
                # Only call .to() if device_map wasn't used
                if not torch.cuda.is_available():
                    model = model.to(device)
                
                print(f"✅ Model loaded from: {path}")
                return model, tokenizer, device
                
            except Exception as e:
                print(f"❌ Error loading from {path}: {e}")
                continue
    
    # Fallback
    print("⚠️ Using base T5 model as fallback")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = model.to(device)
    return model, tokenizer, device
#Hugging Face Spaces optimized loading

def safe_medical_generation(question, model, tokenizer, device):
    """Generate safe medical responses with domain checking"""
    
    # Check for greeting
    if question.lower().strip() in ['hi', 'hello', 'hey', 'greetings']:
        return "Hello! 👋 Welcome to the Malnutrition Medical Assistant. I'm here to help you with questions about severe malnutrition treatment based on WHO guidelines. How can I assist you today?", "answered"
    
    domain_guard = MedicalDomainGuard()
    is_in_domain, reason = domain_guard.is_in_domain(question)
    
    if not is_in_domain:
        if reason.startswith("dangerous_medical"):
            danger_type = reason.split(":")[1]
            return f"❌ I cannot provide advice about {danger_type}. This is outside my specialized domain of severe malnutrition treatment. Please consult a healthcare professional for medical emergencies.", "blocked"
        elif reason == "non_medical":
            return "❌ I'm a medical assistant specialized in severe malnutrition treatment. I can only answer questions about malnutrition, therapeutic diets, and related medical topics from WHO guidelines.", "blocked"
        else:  # uncertain_medical
            return "🔍 I'm not sure if I can answer this accurately. I'm specialized in severe malnutrition treatment. Please ask me about specific malnutrition-related topics.", "blocked"
    
    # If in domain, generate response
    input_text = f"medical information: {question}"
    
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=256,
        truncation=True,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=300,
            num_beams=6,
            early_stopping=True,
            repetition_penalty=3.0,
            length_penalty=1.5,
            no_repeat_ngram_size=3,
            do_sample=False,
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer, "answered"

def main():
    # Header
    st.markdown('<div class="main-header">🍏 Malnutrition Medical Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered guidance for severe malnutrition treatment based on WHO guidelines</div>', unsafe_allow_html=True)
    
    # Safety notice
    st.markdown("""
    <div class="safety-alert">
    <strong>⚠️ Important Safety Notice</strong><br>
    This AI assistant specializes in severe malnutrition treatment based on WHO guidelines. 
    For other health concerns, please consult appropriate medical professionals.
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, tokenizer, device = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check if the model files are available.")
        return
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_history.append({
            "question": "Hi",
            "response": "Hello! 👋 Welcome to the Malnutrition Medical Assistant. I'm here to help you with questions about severe malnutrition treatment based on WHO guidelines. How can I assist you today?",
            "type": "answered",
            "timestamp": time.time()
        })
    
    # Main layout
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        st.markdown("### 💬 Chat")
        
        # Display chat history
        for chat in st.session_state.chat_history:
            # User message
            st.markdown(f"""
            <div style="background: #dbeafe; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; text-align: right; color: #1e40af;">
            <strong>You:</strong> {chat['question']}
            </div>
            """, unsafe_allow_html=True)
            
            # Bot response
            if chat['type'] == "answered":
                response_time_text = f" <span style='font-size: 0.85rem; opacity: 0.8;'>({chat.get('response_time', 0):.2f}s)</span>" if 'response_time' in chat else ""
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 12px; padding: 1rem; margin: 0.5rem 0; color: white;">
                <strong>Assistant:</strong>{response_time_text} {chat['response']}
                </div>
                """, unsafe_allow_html=True)
            else:
                response_time_text = f" <span style='font-size: 0.85rem; opacity: 0.8;'>({chat.get('response_time', 0):.2f}s)</span>" if 'response_time' in chat else ""
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); border-radius: 12px; padding: 1rem; margin: 0.5rem 0; color: white;">
                <strong>Assistant:</strong>{response_time_text} {chat['response']}
                </div>
                """, unsafe_allow_html=True)
        
        # Question input
        question = st.text_input(
            "Your Question",
            placeholder="Type your question here...",
            key="question_input",
            label_visibility="collapsed"
        )
        
        # Buttons
        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            send_button = st.button("🚀 Send", type="primary", use_container_width=True)
        with col_btn2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        # Handle send button
        if send_button and question.strip():
            # Add user question immediately
            st.session_state.chat_history.append({
                "question": question,
                "response": "Loading...",
                "type": "loading",
                "timestamp": time.time()
            })
            st.rerun()
        
        # Process loading state
        if st.session_state.chat_history and st.session_state.chat_history[-1]['type'] == 'loading':
            last_question = st.session_state.chat_history[-1]['question']
            with st.spinner("🔍 Analyzing your question..."):
                start_time = time.time()
                response, response_type = safe_medical_generation(last_question, model, tokenizer, device)
                response_time = time.time() - start_time
            
            st.session_state.chat_history[-1] = {
                "question": last_question,
                "response": response,
                "type": response_type,
                "timestamp": time.time(),
                "response_time": response_time
            }
            st.rerun()
        
        # Handle clear button
        if clear_button:
            st.session_state.chat_history = []
            st.session_state.chat_history.append({
                "question": "Hi",
                "response": "Hello! 👋 Welcome to the Malnutrition Medical Assistant. I'm here to help you with questions about severe malnutrition treatment based on WHO guidelines. How can I assist you today?",
                "type": "answered",
                "timestamp": time.time()
            })
            st.rerun()
    
    with col2:
        st.markdown("### 📋 Sample Questions")
        
        sample_questions = {
            "✅ In-Domain": [
                "What is severe malnutrition?",
                "How should I treat hypoglycemia?",
                "What is the F-75 diet?",
                "How do I recognize dehydration?",
                "What is ReSoMal?",
                "When should I give a blood transfusion?",
                "How often should I feed a malnourished child?"
            ],
            "🚫 Out-of-Domain": [
                "How do I change a car tire?",
                "What is the treatment for diabetes?",
                "How to perform CPR?",
                "What is the capital of France?",
                "How to bake a cake?"
            ]
        }
        
        for category, questions in sample_questions.items():
            st.markdown(f"<p style='color: #1f2937; font-weight: 600; margin-bottom: 0.5rem;'>{category}</p>", unsafe_allow_html=True)
            for q in questions:
                if st.button(q, key=f"sample_{q}", use_container_width=True):
                    st.session_state.chat_history.append({
                        "question": q,
                        "response": "Loading...",
                        "type": "loading",
                        "timestamp": time.time()
                    })
                    st.rerun()
    
    # Footer with domain information
    st.markdown("---")
    st.markdown("""
    <div style="margin-top: 3rem; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);">
        <h2 style="text-align: center; color: white; margin-bottom: 2rem; font-size: 1.8rem;">🎯 About This Assistant</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; padding: 0 2rem;">
            <div style="background: rgba(255, 255, 255, 0.95); border-radius: 16px; padding: 2rem; text-align: center; backdrop-filter: blur(10px);">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🎯</div>
                <div style="font-weight: 700; font-size: 1.1rem; color: #1f2937; margin-bottom: 1rem;">Specialized Domain</div>
                <div style="color: #6b7280; font-size: 0.95rem; line-height: 1.8;">
                    • Severe malnutrition treatment<br>
                    • WHO therapeutic protocols<br>
                    • Medical guidance only
                </div>
            </div>
            <div style="background: rgba(255, 255, 255, 0.95); border-radius: 16px; padding: 2rem; text-align: center; backdrop-filter: blur(10px);">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🛡️</div>
                <div style="font-weight: 700; font-size: 1.1rem; color: #1f2937; margin-bottom: 1rem;">Safety Features</div>
                <div style="color: #6b7280; font-size: 0.95rem; line-height: 1.8;">
                    • Domain boundary enforcement<br>
                    • Dangerous topic blocking<br>
                    • Clear scope communication
                </div>
            </div>
            <div style="background: rgba(255, 255, 255, 0.95); border-radius: 16px; padding: 2rem; text-align: center; backdrop-filter: blur(10px);">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📚</div>
                <div style="font-weight: 700; font-size: 1.1rem; color: #1f2937; margin-bottom: 1rem;">Based On</div>
                <div style="color: #6b7280; font-size: 0.95rem; line-height: 1.8;">
                    • WHO Malnutrition Guidelines<br>
                    • Evidence-based protocols<br>
                    • Clinical best practices
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
