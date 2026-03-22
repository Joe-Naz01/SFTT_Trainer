# Healthcare Chatbot: Fine-Tuning Llama for Medical QA 🩺
This repository contains a comprehensive pipeline for fine-tuning Large Language Models (LLMs) to optimize their performance for specialized healthcare applications. The project focuses on transforming a base Llama model into a specialized medical assistant capable of classifying patient intent and answering clinical queries.

## Project Overview
The core objective is to build a training and evaluation pipeline for a healthcare chatbot used by hospitals for patient onboarding. By fine-tuning Llama models on domain-specific data, we bridge the gap between general-purpose language understanding and specialized medical knowledge.

## Key Features
- **Medical Data Engineering: Automated pipeline to load and preprocess the MedQuad-MedicalQnADataset.

- **Supervised Fine-Tuning (SFT): Implementation of Hugging Face’s SFTTrainer to run efficient fine-tuning loops.

- **Recipe-Based Configuration: Utilizing TorchTune for streamlined, recipe-based task configuration and data preparation.

- **Memory Efficiency: Application of quantization and parameter-efficient techniques (like LoRA) to run large models on consumer-grade hardware.

##  Tech Stack
**Language: Python

**Core Libraries: torch, torchtune, transformers (Hugging Face)

**Data Handling: datasets, pandas, PyYAML

**Optimization: bitsandbytes (for 8-bit quantization)

## Dataset: MedQuad
The model is trained on the MedQuad-MedicalQnADataset, a high-quality collection of medical question-answer pairs. The pipeline includes:

Intent Classification: Categorizing patient queries to ensure accurate routing.

Prompt Engineering: Formatting raw medical data into instruction-based prompts for the LLM.

## Setup & Installation
1. Environment Configuration
It is recommended to use a Conda environment to manage the specific versions of Torch and Transformers required:

Bash
''' 
git clone https://github.com/Joe-Naz01/SFTT_Trainer.git\
cd SFTT_Trainer

conda create -n llama_ft python=3.10 -y
conda activate llama_ft
pip install -r requirements.txt
jupyter notebook
'''

## Skills Demonstrated
**LLM Orchestration: Building end-to-end pipelines from raw data to model evaluation.

**Resource Management: Implementing memory-efficient training strategies for large-scale models.

**Domain Adaptation: Specializing general AI models for high-stakes industries like healthcare.
