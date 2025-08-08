# Resume Matching & Ranking Application

> **Note:**  
> This project uses the **Google Gemini API** to generate summaries for the top 5 matched resumes.  
> I have used **my own API key**, which has a **daily usage limit** — please be mindful of this if you run the app frequently.

## 📌 Overview
This repository contains a **resume–job description matching system** that ranks resumes based on semantic similarity with a provided JD, and optionally generates AI-based fit summaries.

The repo includes:
- **`engine.py`** → Core logic for text extraction, cleaning, chunking, embeddings, FAISS similarity search, and Gemini summaries.
- **`app.py`** → Streamlit UI to upload a JD and multiple resumes, rank them, and display summaries.
- **Root_Code.ipynb** → Experimental / evaluation code from the development phase (derived from the Jupyter notebook).
- **`requirements.txt`** → All Python dependencies.

---

## 🚀 Instructions to Run

### 1. Clone the Repository
```bash
git clone https://github.com/kuldeepbv/Candidate_Recommendation_Engine.git
cd Candidate_Recommendation_Engine
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate    # On macOS/Linux
venv\Scripts\activate       # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Google Gemini API Key
Replace the placeholder in `engine.py` with your own Gemini API key:
```python
GEMINI_API_KEY = "your_api_key_here"
```
Or set it as an environment variable:
```bash
export GEMINI_API_KEY="your_api_key_here"      # macOS/Linux
setx GEMINI_API_KEY "your_api_key_here"        # Windows
```

### 5. Run the Streamlit App
```bash
streamlit run app.py
```
The app will open in your browser (default: `http://localhost:8501`).

---

## 🧠 Approach

1. **Text Extraction**
   - PDFs via `PyMuPDF`
   - DOCX via `python-docx`
   - TXT via standard Python file handling

2. **Text Cleaning & Lemmatization**
   - Lowercasing, removing punctuation/non-alphanumeric characters
   - Removing stopwords
   - Lemmatizing words using NLTK

3. **Chunking**
   - Splits text into overlapping chunks (default: `chunk_size=200`, `overlap=30` words)

4. **Embeddings**
   - Sentence embeddings generated via `all-mpnet-base-v2` from `sentence-transformers`
   - Mean-pooling of chunk embeddings to create a single vector per resume

5. **Similarity Search**
   - Uses FAISS index (Inner Product on normalized vectors = cosine similarity)
   - Computes similarity between JD and each resume

6. **Ranking**
   - Resumes ranked in descending similarity order

7. **AI Fit Summaries**
   - For top 5 resumes, generates 2–3 sentence fit summaries using the Google Gemini API

---

## 📏 Assumptions
- All resumes and the JD are in English.
- Input files are either PDF, DOCX, or TXT.
- For similarity, cosine distance on L2-normalized vectors is sufficient.
- Chosen embedding model (`all-mpnet-base-v2`) provides a good trade-off between accuracy and performance.
- Chunk size of `200` words with `30` word overlap gives optimal coverage for context.
- Gemini summaries are supplementary; the ranking works independently if the API limit is reached.

---

## 📂 Repository Structure
```
Resume_App_try/
│
├── app.py                  # Streamlit app
├── engine.py               # Core logic
├── root_code.py            # Experimental/evaluation code from development notebook
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── data/                   # (Optional) Store FAISS index & metadata here
```

---

## 🛠️ Tech Stack
- **Python 3.8+**
- **Streamlit** for UI
- **FAISS** for vector search
- **Sentence Transformers** for embeddings
- **Google Gemini API** for summaries
- **NLTK** for preprocessing

---

## 📧 Contact
For any issues or questions, feel free to open a GitHub issue or reach out directly.
