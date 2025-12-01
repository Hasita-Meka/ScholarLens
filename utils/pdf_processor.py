"""
PDF Processing utilities for ScholarLens
Handles PDF text extraction, cleaning, and chunking
"""

import re
import pdfplumber
from typing import List, Dict, Tuple, Optional
import io
import logging
import warnings

logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*gray non-stroke color.*")
warnings.filterwarnings("ignore", message=".*FontBBox.*")
warnings.filterwarnings("ignore", message=".*cannot be parsed.*")


def extract_text_from_pdf(pdf_file) -> Dict[str, any]:
    """
    Extract text content from a PDF file
    Returns structured content with sections identified
    """
    full_text = ""
    pages = []
    
    try:
        if hasattr(pdf_file, 'read'):
            pdf_bytes = pdf_file.read()
            pdf_file.seek(0)
            pdf = pdfplumber.open(io.BytesIO(pdf_bytes))
        else:
            pdf = pdfplumber.open(pdf_file)
        
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            pages.append({
                'page_num': i + 1,
                'text': page_text
            })
            full_text += page_text + "\n"
        
        pdf.close()
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'text': '',
            'pages': []
        }
    
    cleaned_text = clean_text(full_text)
    
    sections = identify_sections(cleaned_text)
    
    metadata = extract_metadata(cleaned_text)
    
    return {
        'success': True,
        'text': cleaned_text,
        'raw_text': full_text,
        'pages': pages,
        'sections': sections,
        'metadata': metadata
    }


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing noise
    """
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\(\d{4}\)', lambda m: m.group(0), text)
    
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'(\n\s*){3,}', '\n\n', text)
    
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line and len(line) > 2:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def identify_sections(text: str) -> Dict[str, str]:
    """
    Identify common paper sections from text
    """
    sections = {
        'title': '',
        'abstract': '',
        'introduction': '',
        'methods': '',
        'results': '',
        'discussion': '',
        'conclusion': '',
        'references': ''
    }
    
    section_patterns = {
        'abstract': r'(?:^|\n)\s*(?:abstract|summary)\s*[:\n]?\s*(.*?)(?=\n\s*(?:1\.|I\.|introduction|keywords|1\s+introduction))',
        'introduction': r'(?:^|\n)\s*(?:1\.?\s*)?introduction\s*[:\n]?\s*(.*?)(?=\n\s*(?:2\.|II\.|related|background|method|approach))',
        'methods': r'(?:^|\n)\s*(?:\d\.?\s*)?(?:method|methodology|approach|model)\s*[:\n]?\s*(.*?)(?=\n\s*(?:\d\.|experiment|result|evaluation))',
        'results': r'(?:^|\n)\s*(?:\d\.?\s*)?(?:result|experiment|evaluation)\s*[:\n]?\s*(.*?)(?=\n\s*(?:\d\.|discussion|conclusion|related))',
        'conclusion': r'(?:^|\n)\s*(?:\d\.?\s*)?conclusion\s*[:\n]?\s*(.*?)(?=\n\s*(?:reference|acknowledge|appendix|$))',
    }
    
    text_lower = text.lower()
    
    for section_name, pattern in section_patterns.items():
        match = re.search(pattern, text_lower, re.DOTALL | re.IGNORECASE)
        if match:
            start = match.start(1) if match.lastindex else match.start()
            end = match.end(1) if match.lastindex else match.end()
            sections[section_name] = text[start:end].strip()[:5000]
    
    lines = text.split('\n')
    if lines:
        potential_title = lines[0].strip()
        if len(potential_title) > 10 and len(potential_title) < 300:
            sections['title'] = potential_title
    
    return sections


def extract_metadata(text: str) -> Dict[str, any]:
    """
    Extract metadata from paper text
    """
    metadata = {
        'title': '',
        'authors': [],
        'year': None,
        'doi': None,
        'emails': [],
        'institutions': []
    }
    
    doi_pattern = r'10\.\d{4,}/[^\s]+'
    doi_match = re.search(doi_pattern, text)
    if doi_match:
        metadata['doi'] = doi_match.group(0)
    
    year_pattern = r'\b(19|20)\d{2}\b'
    years = re.findall(year_pattern, text[:2000])
    if years:
        valid_years = [int(y) for y in years if 1990 <= int(y) <= 2025]
        if valid_years:
            metadata['year'] = max(valid_years)
    
    email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
    emails = re.findall(email_pattern, text[:3000])
    metadata['emails'] = list(set(emails))[:10]
    
    lines = text.split('\n')
    if lines:
        metadata['title'] = lines[0].strip()[:500]
    
    return metadata


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """
    Split text into overlapping chunks for RAG
    """
    chunks = []
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current_chunk = ""
    current_start = 0
    
    for i, sentence in enumerate(sentences):
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip():
                chunks.append({
                    'index': len(chunks),
                    'content': current_chunk.strip(),
                    'start_sentence': current_start,
                    'end_sentence': i - 1
                })
            
            overlap_text = ""
            j = i - 1
            while j >= 0 and len(overlap_text) < overlap:
                overlap_text = sentences[j] + " " + overlap_text
                j -= 1
            
            current_chunk = overlap_text + sentence + " "
            current_start = j + 1
    
    if current_chunk.strip():
        chunks.append({
            'index': len(chunks),
            'content': current_chunk.strip(),
            'start_sentence': current_start,
            'end_sentence': len(sentences) - 1
        })
    
    return chunks


def extract_references(text: str) -> List[str]:
    """
    Extract references from paper text
    """
    references = []
    
    ref_section_match = re.search(
        r'(?:references|bibliography)\s*\n(.*?)(?:\n\s*(?:appendix|supplementary)|$)',
        text.lower(),
        re.DOTALL
    )
    
    if ref_section_match:
        ref_text = text[ref_section_match.start(1):ref_section_match.end(1)]
        
        ref_patterns = [
            r'\[\d+\]\s*[^[\]]+',
            r'\d+\.\s*[A-Z][^.]+\.\s*\d{4}\.',
        ]
        
        for pattern in ref_patterns:
            matches = re.findall(pattern, ref_text)
            if matches:
                references.extend([m.strip() for m in matches])
                break
    
    return references[:100]


def get_section_for_chunk(chunk_text: str, sections: Dict[str, str]) -> str:
    """
    Determine which section a chunk belongs to
    """
    chunk_lower = chunk_text.lower()[:200]
    
    for section_name, section_text in sections.items():
        if section_text and chunk_lower in section_text.lower():
            return section_name
    
    if 'abstract' in chunk_lower or 'summary' in chunk_lower:
        return 'abstract'
    elif 'introduction' in chunk_lower:
        return 'introduction'
    elif 'method' in chunk_lower or 'approach' in chunk_lower:
        return 'methods'
    elif 'result' in chunk_lower or 'experiment' in chunk_lower:
        return 'results'
    elif 'conclusion' in chunk_lower:
        return 'conclusion'
    elif 'discussion' in chunk_lower:
        return 'discussion'
    
    return 'body'
