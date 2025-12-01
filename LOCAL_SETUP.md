# ScholarLens - Local Setup Guide

This guide explains how to run ScholarLens on your local machine.

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

## Installation Steps

### 1. Create a Virtual Environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install streamlit sqlalchemy psycopg2-binary openai python-dotenv networkx plotly pandas numpy scikit-learn pdfplumber nltk requests trafilatura
```

### 3. Configure Environment Variables

Create a `.env` file in the project root directory:

```
# For SQLite (easiest - no database setup needed):
DATABASE_URL=sqlite:///scholarlens.db

# For PostgreSQL (if you have it installed):
# DATABASE_URL=postgresql://username:password@localhost:5432/scholarlens

# AI Provider - Choose ONE of the following:

# Option 1: Google AI Studio (FREE - Recommended!)
# Get your free API key at: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=your-google-api-key-here

# Option 2: OpenAI (Paid)
# Get your API key at: https://platform.openai.com/api-keys
# OPENAI_API_KEY=your-openai-api-key-here
```

### 4. Run the Application

```bash
streamlit run app.py --server.port 5000
```

The app will open in your browser at http://localhost:5000

## AI Provider Options

### Google AI Studio (FREE - Recommended)

1. Go to https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API key"
4. Copy the key and add it to your `.env` file as `GOOGLE_API_KEY`

**Benefits:**
- Completely FREE (1 million tokens/minute)
- Uses Gemini 2.0 Flash model
- No credit card required
- Fast and reliable

### OpenAI (Paid)

1. Go to https://platform.openai.com/api-keys
2. Create an account and add billing
3. Generate an API key
4. Add it to your `.env` file as `OPENAI_API_KEY`

**Note:** If both keys are set, Google AI Studio will be used by default.

## Database Options

### SQLite (Easiest)
- No installation needed
- Just set `DATABASE_URL=sqlite:///scholarlens.db` in your `.env` file
- Data is stored in a local file called `scholarlens.db`

### PostgreSQL (Production-grade)
1. Install PostgreSQL on your machine
2. Create a database: `createdb scholarlens`
3. Set the DATABASE_URL in your `.env` file

## Troubleshooting

### "DATABASE_URL not set" Error
- Make sure you have a `.env` file with the DATABASE_URL variable
- Make sure python-dotenv is installed: `pip install python-dotenv`

### "AI API not configured" Error
- Add either `GOOGLE_API_KEY` or `OPENAI_API_KEY` to your `.env` file
- Google AI Studio is free: https://aistudio.google.com/app/apikey

### SQLAlchemy Errors
- Make sure all packages are installed correctly
- Try reinstalling: `pip install --upgrade sqlalchemy psycopg2-binary`

## Features That Work Without AI API

- PDF Upload and text extraction
- Search across papers
- arXiv and PubMed paper fetching
- Knowledge graph visualization
- Analytics dashboard
- Topic clustering
- Reading list and notes
- Export to Markdown, LaTeX, BibTeX

## Features That Require AI API (Google or OpenAI)

- AI-powered Q&A
- Multi-audience summaries (Expert, Student, Policymaker)
- Flashcard generation
- Quiz generation
- Policy brief generation
- Cross-domain analogies
- Key insights extraction
