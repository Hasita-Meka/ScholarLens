# ScholarLens - AI-Powered Research Intelligence Platform

## Overview
ScholarLens is a comprehensive research intelligence platform that ingests research papers, builds dynamic knowledge graphs, and provides multi-audience explanations, trend forecasts, and interactive exploration tools. It serves as a full research intelligence system that organizes, explains, predicts, and teaches.

## Tech Stack
- **Frontend/Backend**: Streamlit (Python)
- **Database**: PostgreSQL with SQLAlchemy ORM
- **AI/NLP**: Google AI Studio (Gemini) or OpenAI for summaries, Q&A, and content generation
- **Visualization**: Plotly, NetworkX
- **PDF Processing**: pdfplumber
- **Search**: TF-IDF based semantic search with scikit-learn
- **APIs**: arXiv API, PubMed API for paper fetching

## Project Structure
```
/
├── app.py                    # Main Streamlit application
├── models.py                 # SQLAlchemy database models
├── utils/
│   ├── __init__.py
│   ├── pdf_processor.py      # PDF text extraction and processing
│   ├── ner_extractor.py      # Named Entity Recognition for methods/datasets
│   ├── openai_helper.py      # OpenAI API integration
│   ├── semantic_search.py    # TF-IDF based semantic search
│   ├── graph_builder.py      # Knowledge graph construction
│   ├── analytics.py          # SQL analytics and reporting (8+ reports)
│   ├── arxiv_pubmed.py       # arXiv and PubMed API integration
│   ├── topic_modeling.py     # Topic clustering and LDA
│   ├── trend_forecasting.py  # Time-series trend analysis
│   └── export_utils.py       # Export to Markdown, LaTeX, BibTeX, CSV
├── .streamlit/
│   └── config.toml           # Streamlit configuration
└── attached_assets/          # Uploaded files
```

## Key Features

### 1. Smart Corpus Management
- Upload multiple PDF research papers
- Automatic text extraction and entity recognition
- Semantic and entity-based search
- **arXiv and PubMed API integration** for fetching open-access papers
- **Auto-clustering of papers by topic** using TF-IDF and K-means

### 2. Knowledge Graph Explorer
- Interactive visualization of papers, methods, datasets, and authors
- Concept dependency maps (DAGs) showing method prerequisites
- Co-authorship network analysis
- Collaboration opportunity detection

### 3. Evidence-Backed Q&A
- RAG-powered question answering with OpenAI
- Source citations for every answer
- Saved query history

### 4. Multi-Audience Summaries
- Expert (technical) summaries
- Student (with analogies) summaries
- Policymaker (applications & risks) summaries
- **Policy brief generation** with structured reports
- **Cross-domain analogy generation**

### 5. Analytics Dashboard
8+ SQL analytical reports:
1. Top co-authorship pairs by publication count
2. Trending topics over time
3. Papers per institution
4. Research growth by field
5. Top authors by publication frequency
6. Most used datasets
7. Collaboration network density
8. Emerging methods in multiple domains

**Trend Forecasting Features:**
- Time-series analysis of method/dataset popularity
- Identification of emerging and declining methods
- Predictive trajectory visualization

### 6. Learning Mode
- Key insights extraction
- **Flashcard generation** from paper content
- **Quiz generation** with multiple choice questions
- Personalized study roadmaps based on method prerequisites

### 7. Research Workspace
- Reading lists with priority and status tracking
- Note-taking on papers
- **Literature review export** in multiple formats:
  - Markdown
  - LaTeX
  - BibTeX
  - Plain Text
  - CSV

## Database Schema

### Core Tables
- **papers**: Research papers with title, abstract, content, year, DOI, source
- **authors**: Author names with h-index and citations
- **institutions**: Research institutions with type and location
- **methods**: Research methods/techniques with category and usage count
- **datasets**: Datasets with domain and usage count

### Relationship Tables
- **paper_authors**: Many-to-many between papers and authors
- **paper_methods**: Many-to-many between papers and methods
- **paper_datasets**: Many-to-many between papers and datasets
- **author_institutions**: Many-to-many between authors and institutions
- **method_prerequisites**: Self-referential for method dependencies

### User Content Tables
- **paper_chunks**: Text chunks for RAG (semantic search)
- **flashcards**: Generated flashcards for learning
- **notes**: User notes on papers
- **saved_queries**: Saved Q&A history
- **reading_list**: User's reading list with priorities

## Environment Variables Required
- `DATABASE_URL`: PostgreSQL connection string (auto-configured)
- `GOOGLE_API_KEY`: Google AI Studio API key (FREE - recommended)
- `OPENAI_API_KEY`: OpenAI API key (alternative, paid)

Note: If both keys are set, Google AI Studio is used by default.

## Running the Application
```bash
streamlit run app.py --server.port 5000
```

## API Integrations

### arXiv API
- Search papers by query and category
- Fetch paper metadata (title, abstract, authors, year)
- Support for all arXiv categories (cs.AI, cs.LG, cs.CL, etc.)

### PubMed API
- Search biomedical research papers
- Fetch abstracts and metadata
- MeSH term extraction

## Recent Changes
- November 2025: Initial full implementation
- All 12 core MVP features implemented
- 8+ SQL analytical reports added
- arXiv/PubMed API integration
- Topic clustering with TF-IDF + K-means
- Trend forecasting with linear regression
- Multi-format export (Markdown, LaTeX, BibTeX, CSV)
- Interactive knowledge graph visualization
- RAG-powered Q&A system with OpenAI

## Development Notes
- Server binds to 0.0.0.0:5000
- Uses st.rerun() instead of deprecated experimental_rerun
- PostgreSQL database with SQLAlchemy ORM
- No Docker/containerization (Nix environment)
- AI features support Google AI Studio (free) or OpenAI (paid)
- Google AI Studio uses Gemini 2.0 Flash model
- OpenAI uses GPT-4o-mini model
