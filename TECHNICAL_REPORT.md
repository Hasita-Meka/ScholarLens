# ScholarLens - Technical Report

## AI-Powered Research Intelligence Platform

---

## 1. Executive Summary

ScholarLens is a comprehensive research intelligence platform designed to transform how researchers, students, and policymakers interact with academic literature. By leveraging artificial intelligence, natural language processing, and knowledge graph technologies, ScholarLens ingests research papers, builds dynamic knowledge graphs, and provides multi-audience explanations, trend forecasts, and interactive exploration tools.

**Tagline:** *Making Research Intuitive, Interactive, and Insightful*

---

## 2. Purpose and Vision

### 2.1 Problem Statement

Researchers face significant challenges in managing and synthesizing the ever-growing volume of academic literature:
- **Information Overload**: Thousands of papers published daily across disciplines
- **Disconnected Knowledge**: Difficulty identifying relationships between methods, datasets, and research groups
- **Accessibility Barriers**: Technical jargon limits understanding across expertise levels
- **Manual Effort**: Time-consuming literature reviews and knowledge organization
- **Trend Blindness**: Difficulty identifying emerging research directions

### 2.2 Solution

ScholarLens addresses these challenges by providing:
- **Automated Paper Ingestion**: PDF upload, arXiv/PubMed API integration
- **Intelligent Entity Extraction**: Automatic identification of methods, datasets, authors
- **Knowledge Graph Visualization**: Interactive exploration of research connections
- **AI-Powered Summaries**: Multi-audience explanations (expert, student, policymaker)
- **Predictive Analytics**: Trend forecasting and emerging method detection
- **Learning Tools**: Flashcards, quizzes, and study roadmaps

---

## 3. Target Audience

| Audience | Primary Use Cases |
|----------|-------------------|
| **Researchers** | Literature review, trend analysis, collaboration discovery |
| **Graduate Students** | Learning new fields, understanding complex papers, exam preparation |
| **Policymakers** | Understanding research implications, risk assessment |
| **Research Librarians** | Corpus management, research support |
| **Industry R&D Teams** | Technology scouting, competitive analysis |

---

## 4. Core Features

### 4.1 Smart Corpus Management
- **PDF Upload**: Batch upload of research papers with automatic text extraction
- **API Integration**: Search and import papers from arXiv and PubMed
- **Entity Recognition**: Automatic extraction of methods, datasets, and author information
- **Semantic Search**: TF-IDF based concept search across all papers
- **Topic Clustering**: Automatic grouping of papers using K-means clustering

### 4.2 Knowledge Graph Explorer
- **Interactive Visualization**: Plotly-powered graph exploration
- **Multiple View Types**:
  - Paper-Method-Dataset relationships
  - Co-authorship networks
  - Method prerequisite chains (DAGs)
- **Collaboration Detection**: Identify potential research collaborators
- **Entity Statistics**: Usage counts, category distributions

### 4.3 Evidence-Backed Q&A (RAG System)
- **Retrieval-Augmented Generation**: Answers grounded in uploaded papers
- **Source Citations**: Every answer includes paper references
- **Query History**: Saved questions and answers for future reference
- **Context-Aware**: Uses semantic chunking for relevant passage retrieval

### 4.4 Multi-Audience Summaries
- **Expert Summaries**: Technical depth with methodology focus
- **Student Summaries**: Simplified explanations with analogies
- **Policymaker Summaries**: Applications, risks, and societal implications
- **Policy Brief Generation**: Structured reports for decision-makers
- **Cross-Domain Analogies**: Explain concepts using familiar domains

### 4.5 Analytics Dashboard (8+ SQL Reports)

| Report | Description |
|--------|-------------|
| Top Co-Authorship Pairs | Most frequent collaborating author pairs |
| Trending Topics Over Time | Publication trends by year and topic |
| Papers Per Institution | Research output by organization |
| Research Growth by Field | Year-over-year growth analysis |
| Top Authors by Frequency | Most prolific researchers |
| Most Used Datasets | Popular datasets across papers |
| Collaboration Network Density | Network connectivity metrics |
| Emerging Methods | New techniques appearing across domains |

**Trend Forecasting Features:**
- Time-series analysis of method/dataset popularity
- Linear regression for trajectory prediction
- Emerging vs. declining method classification

### 4.6 Learning Mode
- **Key Insights**: Automated extraction of main findings
- **Flashcard Generation**: AI-generated study cards from paper content
- **Quiz Generation**: Multiple-choice questions for self-assessment
- **Study Roadmaps**: Personalized learning paths based on prerequisites

### 4.7 Research Workspace
- **Reading Lists**: Organize papers with priority and status tracking
- **Note-Taking**: Attach notes to individual papers
- **Multi-Format Export**:
  - Markdown (for documentation)
  - LaTeX (for academic writing)
  - BibTeX (for citation managers)
  - CSV (for data analysis)
  - Plain Text

---

## 5. Detailed Implementation Guide

This section provides a comprehensive breakdown of how each feature was built, including the specific libraries, algorithms, and code patterns used.

### 5.1 PDF Processing and Text Extraction

**Library Used:** `pdfplumber`

**Implementation Details:**
```python
# utils/pdf_processor.py
import pdfplumber

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    
    # Section detection using regex patterns
    sections = {
        'abstract': extract_section(text, r'abstract', r'introduction|keywords'),
        'introduction': extract_section(text, r'introduction', r'method|related'),
        'methodology': extract_section(text, r'method', r'result|experiment'),
        'results': extract_section(text, r'result', r'discussion|conclusion'),
        'conclusion': extract_section(text, r'conclusion', r'reference|acknowledgment')
    }
    return {'text': text, 'sections': sections, 'metadata': extract_metadata(pdf)}
```

**Why pdfplumber?**
- Better text extraction than PyPDF2 for academic papers
- Handles multi-column layouts common in research papers
- Preserves reading order and formatting

---

### 5.2 Named Entity Recognition (NER)

**Library Used:** Custom regex-based extractor (no external NLP libraries)

**Implementation Details:**
```python
# utils/ner_extractor.py
import re

# Predefined vocabulary of known methods and datasets
KNOWN_METHODS = [
    'neural network', 'deep learning', 'transformer', 'attention mechanism',
    'BERT', 'GPT', 'CNN', 'RNN', 'LSTM', 'GAN', 'VAE', 'reinforcement learning',
    'random forest', 'SVM', 'gradient descent', 'backpropagation', 'dropout',
    'batch normalization', 'adam optimizer', 'cross-entropy', 'softmax'
]

KNOWN_DATASETS = [
    'ImageNet', 'COCO', 'MNIST', 'CIFAR-10', 'CIFAR-100', 'WikiText',
    'SQuAD', 'GLUE', 'SuperGLUE', 'Penn Treebank', 'WMT', 'LibriSpeech'
]

def extract_entities(text):
    methods = []
    datasets = []
    
    # Pattern matching for known entities
    for method in KNOWN_METHODS:
        if re.search(rf'\b{re.escape(method)}\b', text, re.IGNORECASE):
            methods.append({'name': method, 'category': categorize_method(method)})
    
    # Regex patterns for unknown methods (e.g., "X algorithm", "Y model")
    method_patterns = [
        r'(\w+)\s+algorithm',
        r'(\w+)\s+model',
        r'(\w+)\s+network',
        r'(\w+)\s+method'
    ]
    
    for pattern in method_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if match.lower() not in ['the', 'a', 'an', 'our', 'this']:
                methods.append({'name': match, 'category': 'extracted'})
    
    return {'methods': methods, 'datasets': datasets}
```

**Why Custom Regex Instead of spaCy/NLTK?**
- Lighter weight (no large model downloads)
- Domain-specific vocabulary for academic papers
- Faster processing for batch uploads
- More control over entity categories

---

### 5.3 Semantic Search with TF-IDF

**Library Used:** `scikit-learn` (TfidfVectorizer, cosine_similarity)

**Implementation Details:**
```python
# utils/semantic_search.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticSearchIndex:
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.documents = []
    
    def build_index(self, documents):
        self.documents = documents
        texts = [doc.get('content', '') + ' ' + doc.get('title', '') for doc in documents]
        
        # Adaptive max_df based on corpus size
        n_docs = len(texts)
        max_df = 0.95 if n_docs > 10 else 1.0
        min_df = 2 if n_docs > 5 else 1
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            max_df=max_df,
            min_df=min_df
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
    
    def search(self, query, top_k=5):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum threshold
                results.append({
                    'document': self.documents[idx],
                    'score': float(similarities[idx])
                })
        return results
```

**Why TF-IDF Instead of Embeddings?**
- No API costs (unlike OpenAI embeddings)
- Works offline
- Fast indexing and search
- Sufficient accuracy for academic text
- Adaptive parameters handle small corpora

---

### 5.4 Topic Clustering

**Library Used:** `scikit-learn` (KMeans, TfidfVectorizer)

**Implementation Details:**
```python
# utils/topic_modeling.py
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def cluster_papers(papers, max_clusters=10):
    texts = [p['title'] + ' ' + p.get('abstract', '') for p in papers]
    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Find optimal number of clusters using silhouette score
    best_k = 2
    best_score = -1
    
    for k in range(2, min(max_clusters, len(papers))):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(tfidf_matrix)
        score = silhouette_score(tfidf_matrix, labels)
        
        if score > best_score:
            best_score = score
            best_k = k
    
    # Final clustering with optimal k
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    assignments = kmeans.fit_predict(tfidf_matrix)
    
    # Extract top keywords per cluster
    feature_names = vectorizer.get_feature_names_out()
    keywords = {}
    for i in range(best_k):
        center = kmeans.cluster_centers_[i]
        top_indices = center.argsort()[-5:][::-1]
        keywords[i] = [feature_names[idx] for idx in top_indices]
    
    return {
        'assignments': assignments.tolist(),
        'n_clusters': best_k,
        'keywords': keywords,
        'quality_score': best_score
    }
```

**Algorithm Choice:**
- K-Means for simplicity and speed
- Silhouette score for automatic cluster count selection
- TF-IDF features capture topic-specific vocabulary

---

### 5.5 Knowledge Graph Construction

**Library Used:** `NetworkX` (graph operations), `Plotly` (visualization)

**Implementation Details:**
```python
# utils/graph_builder.py
import networkx as nx
import plotly.graph_objects as go

def build_knowledge_graph(papers, methods, datasets, authors):
    G = nx.Graph()
    
    # Add nodes with types
    for paper in papers:
        G.add_node(f"paper_{paper.id}", 
                   type='paper', 
                   label=paper.title[:30],
                   size=20)
    
    for method in methods:
        G.add_node(f"method_{method.id}",
                   type='method',
                   label=method.name,
                   size=15)
    
    # Add edges based on relationships
    for paper in papers:
        for method in paper.methods:
            G.add_edge(f"paper_{paper.id}", f"method_{method.id}")
        for author in paper.authors:
            G.add_edge(f"paper_{paper.id}", f"author_{author.id}")
    
    return G

def visualize_graph(G):
    # Spring layout for node positioning
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Create Plotly traces
    edge_trace = create_edge_trace(G, pos)
    node_trace = create_node_trace(G, pos)
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)
                    ))
    return fig
```

**Graph Types Implemented:**
1. **Paper-Method-Dataset Graph**: Shows which papers use which methods/datasets
2. **Co-Authorship Network**: Connects authors who published together
3. **Method Prerequisites (DAG)**: Directed graph showing method dependencies

---

### 5.6 AI Integration (OpenAI / Google Gemini)

**Libraries Used:** `openai` (OpenAI API), `google-generativeai` (Gemini API)

**Implementation Details:**
```python
# utils/openai_helper.py
import os
from openai import OpenAI

def get_ai_client():
    # Check for available API keys
    if os.environ.get('OPENAI_API_KEY'):
        return 'openai', OpenAI()
    elif os.environ.get('GOOGLE_API_KEY'):
        import google.generativeai as genai
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        return 'gemini', genai.GenerativeModel('gemini-2.0-flash')
    return None, None

def generate_summary(text, audience='expert'):
    provider, client = get_ai_client()
    
    prompts = {
        'expert': "Provide a technical summary focusing on methodology...",
        'student': "Explain this paper simply with analogies...",
        'policymaker': "Summarize applications, risks, and implications..."
    }
    
    if provider == 'openai':
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a research paper analyst."},
                {"role": "user", "content": f"{prompts[audience]}\n\n{text}"}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    
    elif provider == 'gemini':
        response = client.generate_content(f"{prompts[audience]}\n\n{text}")
        return response.text
```

**Dual Provider Support:**
- OpenAI GPT-4o-mini (paid, higher quality)
- Google Gemini 2.0 Flash (free tier available)
- Automatic fallback between providers

---

### 5.7 RAG (Retrieval-Augmented Generation)

**Implementation Details:**
```python
# Chunking strategy for RAG
def chunk_paper(paper_content, chunk_size=500, overlap=100):
    sentences = paper_content.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) > chunk_size:
            chunks.append('. '.join(current_chunk))
            # Keep overlap
            current_chunk = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
            current_length = sum(len(s) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_length += len(sentence)
    
    return chunks

# RAG Query Processing
def answer_question(question, paper_chunks):
    # Step 1: Retrieve relevant chunks using TF-IDF
    relevant_chunks = search_index.search(question, top_k=5)
    
    # Step 2: Build context from retrieved chunks
    context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
    
    # Step 3: Generate answer with citations
    prompt = f"""Based on the following research paper excerpts, answer the question.
    Include citations to specific papers.
    
    Context:
    {context}
    
    Question: {question}
    """
    
    answer = generate_with_ai(prompt)
    
    # Step 4: Add source references
    sources = [chunk['paper_title'] for chunk in relevant_chunks]
    
    return {'answer': answer, 'sources': sources}
```

---

### 5.8 arXiv and PubMed API Integration

**Libraries Used:** `requests`, `xml.etree.ElementTree`

**Implementation Details:**
```python
# utils/arxiv_pubmed.py
import requests
import xml.etree.ElementTree as ET
import time

def search_arxiv(query, max_results=10):
    base_url = "http://export.arxiv.org/api/query"
    params = {
        'search_query': f'all:{query}',
        'start': 0,
        'max_results': max_results,
        'sortBy': 'relevance'
    }
    
    response = requests.get(base_url, params=params)
    root = ET.fromstring(response.content)
    
    papers = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        paper = {
            'title': entry.find('{http://www.w3.org/2005/Atom}title').text,
            'abstract': entry.find('{http://www.w3.org/2005/Atom}summary').text,
            'authors': [author.find('{http://www.w3.org/2005/Atom}name').text 
                       for author in entry.findall('{http://www.w3.org/2005/Atom}author')],
            'arxiv_id': entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1],
            'year': entry.find('{http://www.w3.org/2005/Atom}published').text[:4],
            'source': 'arxiv'
        }
        papers.append(paper)
    
    time.sleep(0.5)  # Rate limiting
    return papers

def search_pubmed(query, max_results=10):
    # Step 1: Search for IDs
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        'db': 'pubmed',
        'term': query,
        'retmax': max_results,
        'retmode': 'json'
    }
    
    search_response = requests.get(search_url, params=search_params).json()
    pmids = search_response['esearchresult']['idlist']
    
    # Step 2: Fetch details for each ID
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = {
        'db': 'pubmed',
        'id': ','.join(pmids),
        'retmode': 'xml'
    }
    
    fetch_response = requests.get(fetch_url, params=fetch_params)
    # Parse XML and extract paper details...
    
    return papers
```

**Rate Limiting:**
- arXiv: 3 requests/second (0.5s delay implemented)
- PubMed: 3 requests/second (10 with API key)

---

### 5.9 Trend Forecasting

**Library Used:** `numpy`, `scipy` (linear regression)

**Implementation Details:**
```python
# utils/trend_forecasting.py
import numpy as np
from scipy import stats

def analyze_method_trends(method_usage_by_year):
    """
    method_usage_by_year: dict {year: count}
    Returns: trend classification and forecast
    """
    years = sorted(method_usage_by_year.keys())
    counts = [method_usage_by_year[y] for y in years]
    
    if len(years) < 3:
        return {'trend': 'insufficient_data', 'slope': 0}
    
    # Linear regression
    x = np.array(range(len(years)))
    y = np.array(counts)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Classify trend
    if slope > 0.5 and p_value < 0.05:
        trend = 'emerging'
    elif slope < -0.5 and p_value < 0.05:
        trend = 'declining'
    else:
        trend = 'stable'
    
    # Forecast next year
    next_year_prediction = intercept + slope * len(years)
    
    return {
        'trend': trend,
        'slope': slope,
        'r_squared': r_value ** 2,
        'prediction': max(0, next_year_prediction),
        'confidence': 1 - p_value
    }
```

---

### 5.10 Multi-Format Export

**Implementation Details:**
```python
# utils/export_utils.py

def export_to_markdown(papers):
    md = "# Literature Review\n\n"
    for paper in papers:
        md += f"## {paper.title}\n\n"
        md += f"**Authors:** {', '.join([a.name for a in paper.authors])}\n\n"
        md += f"**Year:** {paper.year}\n\n"
        md += f"**Abstract:** {paper.abstract}\n\n"
        md += "---\n\n"
    return md

def export_to_latex(papers):
    latex = "\\documentclass{article}\n\\begin{document}\n\n"
    latex += "\\section{Literature Review}\n\n"
    for paper in papers:
        latex += f"\\subsection{{{escape_latex(paper.title)}}}\n\n"
        latex += f"\\textbf{{Authors:}} {', '.join([a.name for a in paper.authors])}\n\n"
        latex += f"\\textbf{{Year:}} {paper.year}\n\n"
        latex += f"{escape_latex(paper.abstract)}\n\n"
    latex += "\\end{document}"
    return latex

def export_to_bibtex(papers):
    bibtex = ""
    for paper in papers:
        key = generate_citation_key(paper)
        bibtex += f"@article{{{key},\n"
        bibtex += f"  title = {{{paper.title}}},\n"
        bibtex += f"  author = {{{' and '.join([a.name for a in paper.authors])}}},\n"
        bibtex += f"  year = {{{paper.year}}},\n"
        if paper.doi:
            bibtex += f"  doi = {{{paper.doi}}},\n"
        bibtex += "}\n\n"
    return bibtex

def export_to_csv(papers):
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Title', 'Authors', 'Year', 'Abstract', 'DOI', 'Methods'])
    
    for paper in papers:
        writer.writerow([
            paper.title,
            '; '.join([a.name for a in paper.authors]),
            paper.year,
            paper.abstract,
            paper.doi or '',
            '; '.join([m.name for m in paper.methods])
        ])
    
    return output.getvalue()
```

---

### 5.11 Database Design with SQLAlchemy

**Library Used:** `SQLAlchemy` ORM, `PostgreSQL` (production), `SQLite` (fallback)

**Implementation Details:**
```python
# models.py
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Table
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

Base = declarative_base()

# Many-to-Many association tables
paper_authors = Table('paper_authors', Base.metadata,
    Column('paper_id', Integer, ForeignKey('papers.id')),
    Column('author_id', Integer, ForeignKey('authors.id'))
)

paper_methods = Table('paper_methods', Base.metadata,
    Column('paper_id', Integer, ForeignKey('papers.id')),
    Column('method_id', Integer, ForeignKey('methods.id'))
)

class Paper(Base):
    __tablename__ = 'papers'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False)
    abstract = Column(Text)
    content = Column(Text)
    year = Column(Integer)
    doi = Column(String(100))
    source = Column(String(50))  # 'pdf', 'arxiv', 'pubmed'
    source_id = Column(String(100))  # arxiv_id or pmid
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    authors = relationship('Author', secondary=paper_authors, back_populates='papers')
    methods = relationship('Method', secondary=paper_methods, back_populates='papers')
    datasets = relationship('Dataset', secondary=paper_datasets, back_populates='papers')
    chunks = relationship('PaperChunk', back_populates='paper')

# Database connection with pooling
def get_engine():
    database_url = os.environ.get('DATABASE_URL')
    
    if database_url:
        # PostgreSQL with connection pooling
        engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True  # Auto-reconnect on stale connections
        )
    else:
        # SQLite fallback for local development
        engine = create_engine('sqlite:///scholarlens.db')
    
    return engine
```

**Connection Handling:**
- Pool pre-ping for automatic reconnection
- Session-per-request pattern
- Proper session cleanup to prevent leaks

---

### 5.12 Streamlit UI Components

**Key Patterns Used:**
```python
# Session State Management
if 'search_index' not in st.session_state:
    st.session_state.search_index = SemanticSearchIndex()

# Multi-column layouts
col1, col2 = st.columns([3, 1])
with col1:
    st.text_input("Search...")
with col2:
    st.button("Submit")

# Expanders for collapsible content
with st.expander("Paper Details"):
    st.write(paper.abstract)

# Progress indicators
progress_bar = st.progress(0)
for i, item in enumerate(items):
    process(item)
    progress_bar.progress((i + 1) / len(items))

# File downloads
st.download_button(
    label="Download Report",
    data=export_data,
    file_name="report.md",
    mime="text/markdown"
)
```

---

## 6. Technical Architecture

### 5.1 Technology Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit (Python) |
| **Backend** | Python 3.11 |
| **Database** | PostgreSQL with SQLAlchemy ORM |
| **AI/NLP** | OpenAI GPT-4o-mini / Google Gemini 2.0 Flash |
| **Visualization** | Plotly, NetworkX |
| **PDF Processing** | pdfplumber |
| **Search** | TF-IDF with scikit-learn |
| **APIs** | arXiv API, PubMed E-utilities |

### 5.2 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        ScholarLens                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Streamlit  │  │   AI/NLP     │  │  Visualization│          │
│  │   Frontend   │  │   Engine     │  │    Engine     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│  ┌──────┴─────────────────┴─────────────────┴───────┐          │
│  │              Application Core (Python)            │          │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │          │
│  │  │ PDF     │ │ NER     │ │ Semantic│ │ Graph   │ │          │
│  │  │Processor│ │Extractor│ │ Search  │ │ Builder │ │          │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ │          │
│  └──────────────────────┬───────────────────────────┘          │
│                         │                                       │
│  ┌──────────────────────┴───────────────────────────┐          │
│  │           PostgreSQL Database (SQLAlchemy)        │          │
│  │  Papers │ Authors │ Methods │ Datasets │ Notes   │          │
│  └──────────────────────────────────────────────────┘          │
│                         │                                       │
│  ┌──────────────────────┴───────────────────────────┐          │
│  │              External APIs                        │          │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────────┐   │          │
│  │  │ arXiv   │  │ PubMed  │  │ OpenAI/Gemini   │   │          │
│  │  └─────────┘  └─────────┘  └─────────────────┘   │          │
│  └──────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Database Schema

**Core Tables:**
- `papers` - Research papers (title, abstract, content, year, DOI, source)
- `authors` - Author profiles (name, h-index, citations)
- `institutions` - Research organizations (name, type, location)
- `methods` - Research methods/techniques (name, category, usage_count)
- `datasets` - Datasets (name, domain, usage_count)

**Relationship Tables:**
- `paper_authors` - Paper-Author associations
- `paper_methods` - Paper-Method associations
- `paper_datasets` - Paper-Dataset associations
- `author_institutions` - Author-Institution affiliations
- `method_prerequisites` - Method dependency chains

**User Content Tables:**
- `paper_chunks` - Text chunks for semantic search/RAG
- `flashcards` - Generated study flashcards
- `notes` - User notes on papers
- `saved_queries` - Q&A history
- `reading_list` - User reading lists

### 5.4 Module Structure

```
/
├── app.py                    # Main Streamlit application (1300+ lines)
├── models.py                 # SQLAlchemy database models
├── utils/
│   ├── __init__.py
│   ├── pdf_processor.py      # PDF text extraction and section parsing
│   ├── ner_extractor.py      # Named Entity Recognition (regex-based)
│   ├── openai_helper.py      # OpenAI/Gemini API integration
│   ├── semantic_search.py    # TF-IDF vectorization and search
│   ├── graph_builder.py      # NetworkX knowledge graph construction
│   ├── analytics.py          # SQL analytics (8+ reports)
│   ├── arxiv_pubmed.py       # External API integration
│   ├── topic_modeling.py     # K-means clustering, LDA
│   ├── trend_forecasting.py  # Time-series analysis
│   └── export_utils.py       # Multi-format export
└── .streamlit/
    └── config.toml           # Streamlit configuration
```

---

## 6. Key Algorithms and Techniques

### 6.1 Named Entity Recognition (NER)
- **Approach**: Regex pattern matching with domain-specific vocabularies
- **Entities Extracted**: Methods, datasets, techniques, algorithms
- **Example Patterns**: 
  - Methods: "neural network", "transformer", "BERT"
  - Datasets: "ImageNet", "COCO", "MNIST"

### 6.2 Semantic Search
- **Algorithm**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Implementation**: scikit-learn TfidfVectorizer
- **Adaptive Configuration**: max_df adjusted based on corpus size
- **Similarity Metric**: Cosine similarity

### 6.3 Topic Clustering
- **Algorithm**: K-means clustering on TF-IDF vectors
- **Optimal K Selection**: Silhouette score analysis
- **Label Generation**: Top keywords per cluster

### 6.4 Knowledge Graph Construction
- **Library**: NetworkX for graph operations
- **Visualization**: Plotly for interactive rendering
- **Node Types**: Papers, Authors, Methods, Datasets
- **Edge Types**: Authorship, Usage, Prerequisites

### 6.5 RAG (Retrieval-Augmented Generation)
- **Chunking Strategy**: Sentence-based with overlap
- **Retrieval**: TF-IDF similarity matching
- **Generation**: OpenAI GPT-4o-mini or Gemini 2.0 Flash
- **Citation Format**: Inline references to source papers

### 6.6 Trend Forecasting
- **Method**: Linear regression on time-series data
- **Features**: Year-over-year usage counts
- **Output**: Trajectory classification (emerging/stable/declining)

---

## 7. AI Integration

### 7.1 Supported Providers

| Provider | Model | Cost |
|----------|-------|------|
| **OpenAI** | GPT-4o-mini | Paid (API usage) |
| **Google AI Studio** | Gemini 2.0 Flash | Free tier available |

### 7.2 AI-Powered Features

1. **Multi-Audience Summarization**
   - Expert: Technical methodology focus
   - Student: Analogies and simplified explanations
   - Policymaker: Applications and risks

2. **Question Answering**
   - Context-aware responses
   - Source citations
   - Follow-up suggestions

3. **Content Generation**
   - Flashcard creation
   - Quiz question generation
   - Policy brief drafting

4. **Cross-Domain Analogies**
   - Explain ML concepts using cooking analogies
   - Translate technical jargon to everyday language

---

## 8. Security and Data Privacy

### 8.1 Security Measures
- **Environment Variables**: Sensitive keys stored in environment secrets
- **Database**: PostgreSQL with connection pooling and SSL
- **No Data Sharing**: User uploads remain private
- **Session Isolation**: Per-user session state management

### 8.2 API Key Management
- Keys stored as environment secrets (not in code)
- Support for multiple AI providers
- Graceful degradation when APIs unavailable

---

## 9. Deployment Options

### 9.1 Cloud Deployment (Replit)
- **Database**: PostgreSQL (Neon-backed)
- **Hosting**: Replit infrastructure
- **URL**: Accessible via Replit domain or custom domain
- **Scaling**: Automatic based on usage

### 9.2 Local Deployment
- **Requirements**: Python 3.11+, PostgreSQL (optional)
- **Fallback**: SQLite for offline use
- **Configuration**: `.env` file with API keys

---

## 10. Performance Considerations

### 10.1 Optimizations
- **Connection Pooling**: SQLAlchemy pool with pre-ping
- **Lazy Loading**: On-demand data fetching
- **Caching**: Session state for expensive computations
- **Batch Processing**: PDF upload batching

### 10.2 Scalability
- **Database**: PostgreSQL supports large datasets
- **Search**: TF-IDF scales to thousands of documents
- **Graph**: NetworkX handles moderate graph sizes

---

## 11. Future Roadmap

### Planned Enhancements
1. **Citation Network Analysis**: Paper citation graphs
2. **Collaborative Filtering**: Personalized paper recommendations
3. **PDF Annotation**: In-document highlighting and notes
4. **Multi-Language Support**: Translation of non-English papers
5. **Mobile Responsive Design**: Improved mobile experience
6. **API Endpoints**: RESTful API for external integration
7. **Advanced NER**: Transformer-based entity extraction
8. **Real-time Collaboration**: Multi-user workspaces

---

## 12. Conclusion

ScholarLens represents a significant advancement in research intelligence tools, combining modern AI capabilities with intuitive user interfaces to democratize access to academic knowledge. By automating tedious tasks like entity extraction, literature organization, and trend analysis, ScholarLens enables researchers to focus on what matters most: advancing human knowledge.

The platform's multi-audience approach ensures that complex research findings can be understood by experts, students, and decision-makers alike, bridging the gap between academic research and real-world application.

---

## Appendix A: Installation Guide

### Cloud (Replit)
1. Fork the project on Replit
2. Set `OPENAI_API_KEY` or `GOOGLE_API_KEY` in Secrets
3. Click "Run" to start the application

### Local Setup
```bash
# Clone repository
git clone <repository-url>
cd scholarlens

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run application
streamlit run app.py --server.port 5000
```

---

## Appendix B: API Reference

### arXiv API
- **Endpoint**: `http://export.arxiv.org/api/query`
- **Parameters**: `search_query`, `max_results`, `start`
- **Rate Limit**: 3 requests per second

### PubMed API
- **Search**: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi`
- **Fetch**: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi`
- **Rate Limit**: 3 requests per second (10 with API key)

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **RAG** | Retrieval-Augmented Generation - AI technique combining search with LLMs |
| **TF-IDF** | Term Frequency-Inverse Document Frequency - Text vectorization method |
| **NER** | Named Entity Recognition - Extracting structured entities from text |
| **Knowledge Graph** | Network representation of entities and relationships |
| **DAG** | Directed Acyclic Graph - Used for prerequisite chains |

---

*Document Version: 1.0*  
*Last Updated: December 2024*  
*Platform: ScholarLens v1.0*
