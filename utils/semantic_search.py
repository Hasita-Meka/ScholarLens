"""
Semantic search utilities for ScholarLens
Provides vector-based search using TF-IDF and cosine similarity
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json


class SemanticSearch:
    """
    Semantic search using TF-IDF vectorization
    Lightweight alternative to heavy embedding models
    """
    
    def __init__(self):
        self.documents = []
        self.document_vectors = None
        self.metadata = []
        self.is_fitted = False
        self.vectorizer = None
        self._create_vectorizer()
    
    def _create_vectorizer(self):
        """Create vectorizer with appropriate settings for document count"""
        doc_count = len(self.documents)
        if doc_count <= 1:
            max_df = 1.0
        elif doc_count <= 5:
            max_df = 1.0
        else:
            max_df = 0.95
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=max_df
        )
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """
        Add documents to the search index
        """
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))
        
        if self.documents:
            try:
                self._create_vectorizer()
                self.document_vectors = self.vectorizer.fit_transform(self.documents)
                self.is_fitted = True
            except ValueError as e:
                if "no terms remain" in str(e).lower() or "max_df" in str(e).lower():
                    self.vectorizer = TfidfVectorizer(
                        max_features=5000,
                        stop_words=None,
                        ngram_range=(1, 1),
                        min_df=1,
                        max_df=1.0
                    )
                    try:
                        self.document_vectors = self.vectorizer.fit_transform(self.documents)
                        self.is_fitted = True
                    except Exception:
                        self.is_fitted = False
                else:
                    self.is_fitted = False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for documents similar to query
        """
        if not self.is_fitted or not self.documents:
            return []
        
        query_vector = self.vectorizer.transform([query])
        
        similarities = cosine_similarity(query_vector, self.document_vectors)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:
                result = {
                    'content': self.documents[idx],
                    'score': float(similarities[idx]),
                    'index': int(idx)
                }
                if idx < len(self.metadata):
                    result.update(self.metadata[idx])
                results.append(result)
        
        return results
    
    def clear(self):
        """
        Clear all documents from the index
        """
        self.documents = []
        self.document_vectors = None
        self.metadata = []
        self.is_fitted = False
    
    def get_document_count(self) -> int:
        """
        Get number of indexed documents
        """
        return len(self.documents)


class PaperSearchIndex:
    """
    Search index specifically for research papers
    Maintains separate indices for different content types
    """
    
    def __init__(self):
        self.chunk_search = SemanticSearch()
        self.title_search = SemanticSearch()
        self.abstract_search = SemanticSearch()
        self.method_search = SemanticSearch()
    
    def index_paper(self, paper_id: int, title: str, abstract: str, 
                    chunks: List[Dict], methods: List[str] = None):
        """
        Index a paper's content for search
        """
        self.title_search.add_documents(
            [title],
            [{'paper_id': paper_id, 'type': 'title'}]
        )
        
        if abstract:
            self.abstract_search.add_documents(
                [abstract],
                [{'paper_id': paper_id, 'type': 'abstract'}]
            )
        
        for chunk in chunks:
            self.chunk_search.add_documents(
                [chunk['content']],
                [{
                    'paper_id': paper_id,
                    'chunk_index': chunk.get('index', 0),
                    'section': chunk.get('section', 'body'),
                    'type': 'chunk'
                }]
            )
        
        if methods:
            for method in methods:
                self.method_search.add_documents(
                    [method],
                    [{'paper_id': paper_id, 'type': 'method'}]
                )
    
    def search_all(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search across all indices and combine results
        """
        results = []
        
        title_results = self.title_search.search(query, top_k=3)
        for r in title_results:
            r['source'] = 'title'
            r['score'] *= 1.5
            results.append(r)
        
        abstract_results = self.abstract_search.search(query, top_k=3)
        for r in abstract_results:
            r['source'] = 'abstract'
            r['score'] *= 1.2
            results.append(r)
        
        chunk_results = self.chunk_search.search(query, top_k=top_k)
        for r in chunk_results:
            r['source'] = 'content'
            results.append(r)
        
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:top_k]
    
    def search_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search only in paper content chunks (for RAG)
        """
        return self.chunk_search.search(query, top_k)
    
    def search_by_method(self, method_name: str, top_k: int = 10) -> List[Dict]:
        """
        Find papers using a specific method
        """
        return self.method_search.search(method_name, top_k)


def compute_document_similarity(doc1: str, doc2: str) -> float:
    """
    Compute similarity between two documents
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        vectors = vectorizer.fit_transform([doc1, doc2])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return float(similarity)
    except:
        return 0.0


def find_similar_papers(query_text: str, papers: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Find papers similar to query text
    """
    if not papers:
        return []
    
    texts = [p.get('abstract', '') or p.get('content', '')[:1000] for p in papers]
    texts.append(query_text)
    
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    try:
        vectors = vectorizer.fit_transform(texts)
        
        query_vector = vectors[-1]
        paper_vectors = vectors[:-1]
        
        similarities = cosine_similarity(query_vector, paper_vectors)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:
                paper = papers[idx].copy()
                paper['similarity_score'] = float(similarities[idx])
                results.append(paper)
        
        return results
    except:
        return papers[:top_k]
