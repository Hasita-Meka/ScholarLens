"""
Export utilities for ScholarLens
Generates literature reviews in multiple formats: Markdown, PDF, Word, LaTeX
"""

from typing import List, Dict, Optional
from datetime import datetime
import io


def generate_markdown_review(papers: List[Dict], notes: Dict[int, List[str]] = None,
                            title: str = "Literature Review") -> str:
    """
    Generate a literature review in Markdown format
    """
    md = f"# {title}\n\n"
    md += f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
    md += f"*Total papers reviewed: {len(papers)}*\n\n"
    md += "---\n\n"
    
    if notes is None:
        notes = {}
    
    for paper in papers:
        md += f"## {paper.get('title', 'Untitled')}\n\n"
        
        if paper.get('year'):
            md += f"**Year:** {paper['year']}\n\n"
        
        if paper.get('authors'):
            if isinstance(paper['authors'], list):
                if isinstance(paper['authors'][0], dict):
                    author_names = [a.get('name', '') for a in paper['authors']]
                else:
                    author_names = paper['authors']
                md += f"**Authors:** {', '.join(author_names)}\n\n"
        
        if paper.get('doi'):
            md += f"**DOI:** {paper['doi']}\n\n"
        
        if paper.get('venue') or paper.get('journal'):
            venue = paper.get('venue') or paper.get('journal')
            md += f"**Published in:** {venue}\n\n"
        
        if paper.get('abstract'):
            md += "### Abstract\n\n"
            md += f"{paper['abstract']}\n\n"
        
        if paper.get('methods'):
            if isinstance(paper['methods'], list):
                if isinstance(paper['methods'][0], dict):
                    method_names = [m.get('name', '') for m in paper['methods']]
                else:
                    method_names = paper['methods']
                md += f"**Methods:** {', '.join(method_names)}\n\n"
        
        if paper.get('datasets'):
            if isinstance(paper['datasets'], list):
                if isinstance(paper['datasets'][0], dict):
                    dataset_names = [d.get('name', '') for d in paper['datasets']]
                else:
                    dataset_names = paper['datasets']
                md += f"**Datasets:** {', '.join(dataset_names)}\n\n"
        
        paper_id = paper.get('id')
        if paper_id and paper_id in notes and notes[paper_id]:
            md += "### Notes\n\n"
            for note in notes[paper_id]:
                md += f"- {note}\n"
            md += "\n"
        
        md += "---\n\n"
    
    return md


def generate_latex_review(papers: List[Dict], notes: Dict[int, List[str]] = None,
                         title: str = "Literature Review") -> str:
    """
    Generate a literature review in LaTeX format
    """
    latex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{hyperref}
\usepackage{natbib}
\usepackage{geometry}
\geometry{margin=1in}

\title{""" + _escape_latex(title) + r"""}
\author{ScholarLens}
\date{""" + datetime.now().strftime('%B %d, %Y') + r"""}

\begin{document}
\maketitle

\section*{Overview}
This literature review covers """ + str(len(papers)) + r""" papers.

"""
    
    if notes is None:
        notes = {}
    
    for i, paper in enumerate(papers, 1):
        title_text = _escape_latex(paper.get('title', 'Untitled'))
        latex += f"\\subsection*{{{i}. {title_text}}}\n\n"
        
        if paper.get('year'):
            latex += f"\\textbf{{Year:}} {paper['year']}\n\n"
        
        if paper.get('authors'):
            if isinstance(paper['authors'], list):
                if isinstance(paper['authors'][0], dict):
                    author_names = [a.get('name', '') for a in paper['authors']]
                else:
                    author_names = paper['authors']
                authors_text = _escape_latex(', '.join(author_names))
                latex += f"\\textbf{{Authors:}} {authors_text}\n\n"
        
        if paper.get('doi'):
            doi = _escape_latex(paper['doi'])
            latex += f"\\textbf{{DOI:}} \\href{{https://doi.org/{doi}}}{{{doi}}}\n\n"
        
        if paper.get('abstract'):
            abstract = _escape_latex(paper['abstract'])
            latex += f"\\textbf{{Abstract:}} {abstract}\n\n"
        
        if paper.get('methods'):
            if isinstance(paper['methods'], list):
                if isinstance(paper['methods'][0], dict):
                    method_names = [m.get('name', '') for m in paper['methods']]
                else:
                    method_names = paper['methods']
                methods_text = _escape_latex(', '.join(method_names))
                latex += f"\\textbf{{Methods:}} {methods_text}\n\n"
        
        paper_id = paper.get('id')
        if paper_id and paper_id in notes and notes[paper_id]:
            latex += "\\textbf{Notes:}\n\\begin{itemize}\n"
            for note in notes[paper_id]:
                note_text = _escape_latex(note)
                latex += f"  \\item {note_text}\n"
            latex += "\\end{itemize}\n\n"
        
        latex += "\\vspace{1em}\n\\hrule\n\\vspace{1em}\n\n"
    
    latex += r"""
\end{document}
"""
    
    return latex


def _escape_latex(text: str) -> str:
    """
    Escape special LaTeX characters
    """
    if not text:
        return ""
    
    replacements = [
        ('\\', r'\textbackslash{}'),
        ('&', r'\&'),
        ('%', r'\%'),
        ('$', r'\$'),
        ('#', r'\#'),
        ('_', r'\_'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('~', r'\textasciitilde{}'),
        ('^', r'\textasciicircum{}'),
    ]
    
    for old, new in replacements:
        text = text.replace(old, new)
    
    return text


def generate_bibtex(papers: List[Dict]) -> str:
    """
    Generate BibTeX entries for papers
    """
    bibtex = ""
    
    for i, paper in enumerate(papers):
        title = paper.get('title', 'Untitled')
        year = paper.get('year', 'n.d.')
        
        if paper.get('authors'):
            if isinstance(paper['authors'], list):
                if isinstance(paper['authors'][0], dict):
                    first_author = paper['authors'][0].get('name', 'Unknown')
                else:
                    first_author = paper['authors'][0]
            else:
                first_author = 'Unknown'
        else:
            first_author = 'Unknown'
        
        key = f"{first_author.split()[-1].lower()}{year}_{i}"
        key = ''.join(c if c.isalnum() or c == '_' else '' for c in key)
        
        bibtex += f"@article{{{key},\n"
        bibtex += f"  title = {{{title}}},\n"
        
        if paper.get('authors'):
            if isinstance(paper['authors'], list):
                if isinstance(paper['authors'][0], dict):
                    author_names = [a.get('name', '') for a in paper['authors']]
                else:
                    author_names = paper['authors']
                bibtex += f"  author = {{{' and '.join(author_names)}}},\n"
        
        bibtex += f"  year = {{{year}}},\n"
        
        if paper.get('doi'):
            bibtex += f"  doi = {{{paper['doi']}}},\n"
        
        if paper.get('venue') or paper.get('journal'):
            venue = paper.get('venue') or paper.get('journal')
            bibtex += f"  journal = {{{venue}}},\n"
        
        if paper.get('abstract'):
            abstract = paper['abstract'].replace('{', '').replace('}', '')
            bibtex += f"  abstract = {{{abstract[:500]}}},\n"
        
        bibtex += "}\n\n"
    
    return bibtex


def generate_plain_text_review(papers: List[Dict], notes: Dict[int, List[str]] = None,
                              title: str = "Literature Review") -> str:
    """
    Generate a simple plain text literature review
    """
    text = f"{title}\n"
    text += "=" * len(title) + "\n\n"
    text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    text += f"Papers reviewed: {len(papers)}\n\n"
    text += "-" * 50 + "\n\n"
    
    if notes is None:
        notes = {}
    
    for i, paper in enumerate(papers, 1):
        text += f"{i}. {paper.get('title', 'Untitled')}\n\n"
        
        if paper.get('year'):
            text += f"   Year: {paper['year']}\n"
        
        if paper.get('authors'):
            if isinstance(paper['authors'], list):
                if isinstance(paper['authors'][0], dict):
                    author_names = [a.get('name', '') for a in paper['authors']]
                else:
                    author_names = paper['authors']
                text += f"   Authors: {', '.join(author_names)}\n"
        
        if paper.get('abstract'):
            text += f"\n   Abstract:\n   {paper['abstract'][:500]}...\n"
        
        paper_id = paper.get('id')
        if paper_id and paper_id in notes and notes[paper_id]:
            text += "\n   Notes:\n"
            for note in notes[paper_id]:
                text += f"   - {note}\n"
        
        text += "\n" + "-" * 50 + "\n\n"
    
    return text


def generate_csv_export(papers: List[Dict]) -> str:
    """
    Generate CSV export of papers
    """
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow(['Title', 'Year', 'Authors', 'Abstract', 'DOI', 'Methods', 'Datasets', 'Source'])
    
    for paper in papers:
        authors = ''
        if paper.get('authors'):
            if isinstance(paper['authors'], list):
                if isinstance(paper['authors'][0], dict):
                    authors = '; '.join([a.get('name', '') for a in paper['authors']])
                else:
                    authors = '; '.join(paper['authors'])
        
        methods = ''
        if paper.get('methods'):
            if isinstance(paper['methods'], list):
                if isinstance(paper['methods'][0], dict):
                    methods = '; '.join([m.get('name', '') for m in paper['methods']])
                else:
                    methods = '; '.join(paper['methods'])
        
        datasets = ''
        if paper.get('datasets'):
            if isinstance(paper['datasets'], list):
                if isinstance(paper['datasets'][0], dict):
                    datasets = '; '.join([d.get('name', '') for d in paper['datasets']])
                else:
                    datasets = '; '.join(paper['datasets'])
        
        writer.writerow([
            paper.get('title', ''),
            paper.get('year', ''),
            authors,
            paper.get('abstract', '')[:500],
            paper.get('doi', ''),
            methods,
            datasets,
            paper.get('source', '')
        ])
    
    return output.getvalue()


def create_summary_statistics(papers: List[Dict]) -> Dict:
    """
    Create summary statistics for export
    """
    from collections import Counter
    
    stats = {
        'total_papers': len(papers),
        'year_range': None,
        'top_authors': [],
        'top_methods': [],
        'top_datasets': [],
        'sources': {}
    }
    
    years = [p.get('year') for p in papers if p.get('year')]
    if years:
        stats['year_range'] = f"{min(years)} - {max(years)}"
    
    all_authors = []
    for paper in papers:
        if paper.get('authors'):
            if isinstance(paper['authors'], list):
                if isinstance(paper['authors'][0], dict):
                    all_authors.extend([a.get('name', '') for a in paper['authors']])
                else:
                    all_authors.extend(paper['authors'])
    stats['top_authors'] = [a[0] for a in Counter(all_authors).most_common(10)]
    
    all_methods = []
    for paper in papers:
        if paper.get('methods'):
            if isinstance(paper['methods'], list):
                if isinstance(paper['methods'][0], dict):
                    all_methods.extend([m.get('name', '') for m in paper['methods']])
                else:
                    all_methods.extend(paper['methods'])
    stats['top_methods'] = [m[0] for m in Counter(all_methods).most_common(10)]
    
    all_datasets = []
    for paper in papers:
        if paper.get('datasets'):
            if isinstance(paper['datasets'], list):
                if isinstance(paper['datasets'][0], dict):
                    all_datasets.extend([d.get('name', '') for d in paper['datasets']])
                else:
                    all_datasets.extend(paper['datasets'])
    stats['top_datasets'] = [d[0] for d in Counter(all_datasets).most_common(10)]
    
    sources = [p.get('source', 'unknown') for p in papers]
    stats['sources'] = dict(Counter(sources))
    
    return stats
