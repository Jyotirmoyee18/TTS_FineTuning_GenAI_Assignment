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



# Task 2: Fine-tuning TTS for a Regional Language

# Fine Tuning TTS Bengali

This model is a fine-tuned version of microsoft/speecht5_tts on an "arif11/Bengali_AI_Speech" dataset.

# Introduction
Text-to-Speech (TTS) synthesis has emerged as a vital technology in our increasingly digital world, serving a wide array of applications from enhancing accessibility to powering virtual assistants. This project centers on fine-tuning Microsoft's SpeechT5 TTS model specifically for Bengali language synthesis. By addressing the need for high-quality speech synthesis systems in Bengali, we aim to create a more inclusive technological landscape that accommodates the linguistic diversity of millions of speakers. This endeavor not only enhances communication but also empowers users with tools that cater to their native language, thereby fostering greater engagement and usability.

# Model Overview
Base Model: Microsoft SpeechT5 (microsoft/speecht5_tts)
Fine-Tuned Model: DeepDiveDev/speecht5_finetuned_Bengali
Task: Text-to-Speech (TTS)
Language: Bengali
Dataset: Your chosen dataset for Bengali TTS (e.g., common voice datasets)

# Training Details
Training Data: Selected dataset for Bengali TTS
Validation Data: Split from the training data (e.g., 20% for validation)
Fine-tuning Steps: 1500
Batch Size: 4 (per device)
Gradient Accumulation Steps: 8
Learning Rate: 1e-4
Warm-up Steps: 100

# Key Enhancements and Improvements
Dataset: Fine-tuned on a curated Bengali dataset to improve model performance on TTS tasks.
Speaker Embeddings: Integrated speaker embeddings to maintain speaker characteristics and variations.
Text Preprocessing: Implemented advanced text preprocessing techniques, including handling of numbers and technical terms.
Training Optimizations: Utilized FP16 training and gradient checkpointing for efficient resource usage during training.
Regular Evaluation: Incorporated frequent evaluations throughout training to monitor model performance and make necessary adjustments.    

# Model Features
Speech Generation: Generates natural-sounding speech from Bengali input text.
Speaker Customization: Supports speaker embeddings for personalized voice output.
Technical Vocabulary Handling: Effectively manages technical terms and abbreviations commonly used in Bengali.
Natural Speech Processing: Converts numbers and technical jargon into a more conversational form for fluent speech synthesis.

# Limitations
Language Limitation: Currently limited to the Bengali language and may not support dialectal variations or other languages.
Voice Quality Variations: The quality of generated speech may vary based on input text and speaker embeddings.
Out-of-Domain Performance: Performance on out-of-domain text, slang, or colloquialisms has not been fully evaluated.

# Ethical Considerations
Potential Misuse: The technology can be misused for generating misleading or deepfake audio.
Bias in Voice Generation: The model may reflect biases present in the training data demographics.

## Usage

You can utilize the fine-tuned model with the Hugging Face Transformers library. Below is an example of how to generate speech from input text in Bengali:

```python
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan

# Load the fine-tuned model and processor
model = SpeechT5ForTextToSpeech.from_pretrained("DeepDiveDev/speecht5_finetuned_Bengali")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Generate speech from input text
input_text = "আপনি কেমন আছেন?"  # Replace with your text
inputs = processor(input_text, return_tensors="pt")
output = model.generate(**inputs)

# Convert output to audio (assuming vocoder is set up)
audio = vocoder(output)

# Play or save audio as needed

# Acknowledgements
Base SpeechT5 model by Microsoft
Dataset providers for Bengali language TTS
Contributions from the PARIMAL intern program at IIT Roorkee
