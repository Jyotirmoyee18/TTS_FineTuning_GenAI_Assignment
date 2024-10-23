# TTS_FineTuning_GenAI_Assignment
Implementation of fine-tuning TTS models for technical vocabulary in English and in Bengali, as part of IIT Roorkee’s GenAI Internship. Includes dataset creation, model fine-tuning, and evaluation using MOS scores. Also explores optimization techniques like quantization for faster inference.
# Fine-tuning TTS for English with a Focus on Technical Vocabulary
# Model Overview
Base Model: Microsoft SpeechT5 (microsoft/speecht5_tts)
Fine-Tuned Model: DeepDiveDev/speecht5_finetuned_English
Task: Text-to-Speech (TTS)
Language: English
Dataset: keithito/lj_speech

# Training Details
Training Data: Train split from the keithito/lj_speech dataset
Validation Data: Test split from the same dataset (20% of total data)
Fine-tuning Steps: 1500
Batch Size: 4 (per device)
Gradient Accumulation Steps: 8
Learning Rate: 1e-4
Warm-up Steps: 100

# Key Differences and Improvements:
Specialized Dataset: Fine-tuned on the keithito/lj_speech dataset to significantly boost performance for English TTS tasks, particularly in technical contexts.
Speaker Adaptation: Integration of speaker embeddings enables personalized voice generation while retaining speaker characteristics.
Advanced Text Processing: Features sophisticated text preprocessing capabilities, including:
Conversion of numbers to word form for more natural speech output
Effective handling of technical vocabulary and abbreviations (e.g., API, CUDA, GPU)
Optimized Training Techniques: Utilized mixed precision training (FP16) and gradient checkpointing for enhanced training efficiency on GPU resources.
Continuous Performance Monitoring: Regular evaluations every 100 steps to track and improve model performance throughout training.

# Model Performance
Evaluation Strategy: Regular evaluation steps
Evaluation Frequency: Every 100 steps
Metric: Not specified (uses greater_is_better=False)

# Model Features
Speech Generation: Capable of generating high-quality speech from textual input.
Voice Customization: Supports speaker embeddings for diverse voice outputs tailored to user preferences.
Technical Vocabulary Handling: Specifically designed to accurately synthesize technical terms and industry-specific language.
Natural Speech Synthesis: Includes functionality for converting numbers to words, contributing to more human-like speech generation.


# Limitations
Language Scope: Currently limited to English language output; additional language support is not available.
Variability in Voice Quality: The quality of generated speech may vary based on the input text and selected speaker embeddings.
Performance on Diverse Inputs: Not evaluated for its effectiveness on out-of-domain text or various accents, which may affect accuracy and naturalness.

# Ethical Considerations
Potential for misuse in creating deepfake audio
Bias in voice generation influenced by the demographics of the training data

# Usage
The model can be used with the Hugging Face Transformers library:
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan

from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan



# Acknowledgements
Base SpeechT5 Model: Developed by Microsoft

Dataset: Keithito LJ Speech

Internship Program: PARIMAL intern program at IIT Roorkee
