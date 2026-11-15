# 🌱 PlantPulse AI
### Intelligent Tomato Disease Detection & Agricultural Knowledge Assistant

**PlantPulse AI** is an AI-powered precision agriculture project that combines deep learning and Retrieval-Augmented Generation (RAG) to help users identify tomato leaf diseases and access relevant agricultural knowledge.

The system uses **EfficientNetB0** for image-based disease classification and a **RAG pipeline** to retrieve information from a curated agricultural knowledge base. A locally running Large Language Model (LLM) generates responses grounded in the retrieved information.

---

## 🎯 Project Overview

Plant diseases can significantly affect crop health and agricultural productivity. Identifying diseases early and accessing reliable management information can help farmers make better decisions.

PlantPulse AI explores how modern AI technologies can support this process through two main capabilities:

- **Disease Detection:** Analyze a tomato leaf image and predict its likely condition.
- **Agricultural Knowledge Assistant:** Ask questions about tomato diseases, symptoms, prevention, and plant care using retrieved agricultural references.

The project is designed as a notebook-based AI system, with the complete workflow demonstrated in Jupyter Notebook.

---

## ✨ Key Features

### 🍃 1. Tomato Disease Classification
- Uses a transfer-learning approach with EfficientNetB0.
- Classifies tomato leaf images into three categories:
  - Early Blight
  - Late Blight
  - Healthy
- Uses a confidence threshold to flag uncertain predictions.

### 🧠 2. Retrieval-Augmented Generation (RAG)
- Retrieves relevant information from a curated agricultural knowledge base.
- Uses Sentence Transformers (`all-MiniLM-L6-v2`) for semantic embeddings.
- Stores and searches document embeddings using ChromaDB.
- Organizes agricultural information into meaningful sections.

### 🤖 3. Local LLM Integration
- Integrates Llama 3.2 1B Instruct through GGUF.
- Uses retrieved agricultural information as context for responses.
- Designed to avoid generating recommendations when sufficient supporting evidence is unavailable.

### 🛡️ 4. Reliability-Aware Workflow
- Checks whether user questions are agriculture-related.
- Identifies the relevant tomato disease or plant-care topic.
- Applies a confidence threshold to image predictions.
- Supports abstention when the prediction or retrieved evidence is insufficient.

---

## 🏗️ System Architecture

```text
                  PlantPulse AI
                        |
             +----------+----------+
             |                     |
        Leaf Image             User Question
             |                     |
             v                     v
       EfficientNetB0       Question Validation
             |                     |
             v                     v
      Disease Prediction     Topic & Intent Detection
             |                     |
             |                     v
             |               Sentence Embeddings
             |                     |
             |                     v
             |                 ChromaDB
             |                     |
             |                     v
             |              Relevant Evidence
             |                     |
             +----------+----------+
                        |
                        v
                 Local LLM
               Llama 3.2 1B
                        |
                        v
              Agricultural Response
                        |
                        v
                Source References
```

---

## 🧰 Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Development Environment | Jupyter Notebook |
| Deep Learning | TensorFlow / Keras |
| Image Classification | EfficientNetB0 |
| Transfer Learning | ImageNet Pretrained Weights |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| RAG Framework | LangChain Text Splitters |
| Local Language Model | Llama 3.2 1B Instruct |
| Model Format | GGUF |
| Image Processing | Pillow |
| Data Analysis & Evaluation | NumPy, Scikit-learn |

---

## 📊 Model Performance

The EfficientNetB0 model was trained using a three-class tomato leaf dataset.

### Dataset Distribution

| Class | Training | Validation | Testing |
|---|---:|---:|---:|
| Early Blight | 700 | 150 | 150 |
| Late Blight | 1,336 | 286 | 287 |
| Healthy | 1,113 | 239 | 239 |
| **Total** | **3,149** | **675** | **676** |

### Test Results

| Metric | Result |
|---|---:|
| Test Accuracy | **98.37%** |
| Test Loss | 0.0499 |
| Macro F1-Score | 98.14% |
| Weighted F1-Score | 98.37% |

**Classification F1-Scores**

| Class | F1-Score |
|---|---:|
| Early Blight | 96.62% |
| Late Blight | 98.43% |
| Healthy | 99.38% |

*Note: These results are from a held-out test split of the selected dataset. They do not establish real-world field accuracy. Further evaluation using images collected under different field conditions is necessary.*

---

## 📚 Agricultural Knowledge Base

PlantPulse currently includes curated knowledge covering four topics:

1. **Early Blight** – Symptoms, causes, management, and prevention.
2. **Late Blight** – Symptoms, disease development, and management.
3. **Healthy Tomato Care** – Watering, sunlight, nutrition, and general plant care.
4. **Tomato Flower Drop** – Environmental stress, pollination, common causes, and management.

The knowledge base is organized into text chunks, converted into semantic embeddings, and stored in ChromaDB for retrieval.

Agricultural information is sourced from university extension services and agricultural organizations.

---

## 📁 Project Structure

```text
PlantPulse-AI/
│
├── dataset/
│   ├── Tomato/
│   └── split/
│       ├── train/
│       ├── validation/
│       └── test/
│
├── notebooks/
│   └── PlantPulse_Model.ipynb
│
├── models/
│   ├── plantpulse_efficientnetb0.keras
│   ├── class_names.json
│   └── Llama-3.2-1B-Instruct-Q5_K_M.gguf
│
├── knowledge_base/
│   ├── early_blight.txt
│   ├── late_blight.txt
│   ├── healthy_tomato.txt
│   └── tomato_flower_drop.txt
│
├── chroma_db/
│
├── results/
│
├── plantpulse_setup.py.txt
│
└── README.md
```

*Large datasets, model weights, GGUF files, and generated vector database files are kept outside the Git repository.*

---

## 🚀 Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/PlantPulse-AI.git
```

### 2. Navigate to the Project

```bash
cd PlantPulse-AI
```

### 3. Set Up the Environment

Create and activate a Conda environment:

```bash
conda create -n plantpulse python=3.12
conda activate plantpulse
```

Install the required packages:

```bash
pip install tensorflow numpy matplotlib pillow scikit-learn
pip install sentence-transformers chromadb
pip install langchain-text-splitters llama-cpp-python
pip install jupyter ipykernel
```

### 4. Prepare Project Assets

Place the required files in their corresponding directories:

- Trained EfficientNetB0 model.
- Class names JSON file.
- Local GGUF language model.
- ChromaDB knowledge collection.
- Required dataset and knowledge-base files.

### 5. Open Jupyter Notebook

```bash
jupyter notebook
```

Open `notebooks/PlantPulse_Model.ipynb` and select the PlantPulse Python environment.

Follow the notebook cells to initialize the models and run the pipeline.

---

## 🔬 Current Scope & Limitations

PlantPulse is currently a research and learning prototype.

- Disease classification is limited to three tomato leaf categories.
- The system does not currently support all crop species or plant diseases.
- The confidence threshold is provisional and requires calibration.
- The dataset is controlled and held-out test results may not represent field conditions.
- LLM responses depend on the quality and relevance of retrieved evidence.
- Insufficient evidence should result in a cautious response rather than an unsupported treatment recommendation.
- Real-world agricultural validation is still required.

**PlantPulse is an educational decision-support prototype, not a replacement for professional agricultural diagnosis.**

---

## 🎓 Project Motivation

This project explores the practical integration of:

- Deep Learning
- Computer Vision
- Transfer Learning
- Natural Language Processing
- Retrieval-Augmented Generation
- Vector Databases
- Local Large Language Models

The goal is to understand how these technologies can work together to build more accessible and evidence-oriented agricultural AI systems.

---

## 👨‍💻 Author

**Muhammad Haneef Ali**

Computer Science Student | AI & Software Development Enthusiast

Interested in Artificial Intelligence, Machine Learning, Deep Learning, and AI Engineering.

---

⭐ If you find this project interesting, consider giving the repository a star!
