"""
Topic modeling and clustering for ScholarLens
Auto-clusters papers by topic using TF-IDF and clustering algorithms
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics import silhouette_score
from collections import Counter
import re


class TopicModeler:
    """
    Topic modeling using Latent Dirichlet Allocation (LDA)
    """
    
    def __init__(self, n_topics: int = 10, max_features: int = 5000):
        self.n_topics = n_topics
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            max_df=0.95,
            min_df=2,
            ngram_range=(1, 2)
        )
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            learning_method='online',
            max_iter=20
        )
        self.is_fitted = False
        self.feature_names = None
    
    def fit(self, documents: List[str]):
        """
        Fit the topic model on documents
        """
        if not documents:
            return
        
        cleaned_docs = [self._preprocess(doc) for doc in documents]
        cleaned_docs = [doc for doc in cleaned_docs if len(doc) > 50]
        
        if len(cleaned_docs) < 2:
            return
        
        tfidf_matrix = self.vectorizer.fit_transform(cleaned_docs)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        self.lda.fit(tfidf_matrix)
        self.is_fitted = True
    
    def get_document_topics(self, documents: List[str]) -> List[List[Tuple[int, float]]]:
        """
        Get topic distribution for each document
        """
        if not self.is_fitted:
            return []
        
        cleaned_docs = [self._preprocess(doc) for doc in documents]
        tfidf_matrix = self.vectorizer.transform(cleaned_docs)
        
        topic_distributions = self.lda.transform(tfidf_matrix)
        
        results = []
        for dist in topic_distributions:
            topics = [(i, float(prob)) for i, prob in enumerate(dist)]
            topics.sort(key=lambda x: x[1], reverse=True)
            results.append(topics[:3])
        
        return results
    
    def get_topic_words(self, n_words: int = 10) -> Dict[int, List[str]]:
        """
        Get top words for each topic
        """
        if not self.is_fitted or self.feature_names is None:
            return {}
        
        topic_words = {}
        for topic_idx, topic in enumerate(self.lda.components_):
            top_word_indices = topic.argsort()[:-n_words-1:-1]
            topic_words[topic_idx] = [self.feature_names[i] for i in top_word_indices]
        
        return topic_words
    
    def get_topic_labels(self) -> Dict[int, str]:
        """
        Generate human-readable labels for topics
        """
        topic_words = self.get_topic_words(n_words=3)
        labels = {}
        
        for topic_id, words in topic_words.items():
            labels[topic_id] = " / ".join(words[:3]).title()
        
        return labels
    
    def _preprocess(self, text: str) -> str:
        """
        Preprocess text for topic modeling
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


class PaperClusterer:
    """
    Clusters papers based on content similarity
    """
    
    def __init__(self, n_clusters: int = 5, method: str = 'kmeans'):
        self.n_clusters = n_clusters
        self.method = method
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            stop_words='english',
            max_df=0.9,
            min_df=1
        )
        self.clusterer = None
        self.is_fitted = False
        self.tfidf_matrix = None
    
    def fit_predict(self, documents: List[str]) -> List[int]:
        """
        Cluster documents and return cluster labels
        """
        if not documents:
            return []
        
        cleaned_docs = [self._preprocess(doc) for doc in documents]
        
        self.tfidf_matrix = self.vectorizer.fit_transform(cleaned_docs)
        
        n_samples = len(documents)
        actual_clusters = min(self.n_clusters, n_samples)
        
        if actual_clusters < 2:
            return [0] * n_samples
        
        if self.method == 'kmeans':
            self.clusterer = KMeans(
                n_clusters=actual_clusters,
                random_state=42,
                n_init=10
            )
        else:
            self.clusterer = AgglomerativeClustering(
                n_clusters=actual_clusters,
                linkage='ward'
            )
        
        if self.method == 'kmeans':
            labels = self.clusterer.fit_predict(self.tfidf_matrix)
        else:
            labels = self.clusterer.fit_predict(self.tfidf_matrix.toarray())
        
        self.is_fitted = True
        return labels.tolist()
    
    def get_cluster_keywords(self, documents: List[str], labels: List[int], 
                            n_keywords: int = 5) -> Dict[int, List[str]]:
        """
        Get representative keywords for each cluster
        """
        if not self.is_fitted:
            return {}
        
        feature_names = self.vectorizer.get_feature_names_out()
        cluster_keywords = {}
        
        for cluster_id in set(labels):
            cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
            
            if not cluster_indices:
                continue
            
            cluster_tfidf = self.tfidf_matrix[cluster_indices].mean(axis=0)
            cluster_tfidf = np.asarray(cluster_tfidf).flatten()
            
            top_indices = cluster_tfidf.argsort()[:-n_keywords-1:-1]
            cluster_keywords[cluster_id] = [feature_names[i] for i in top_indices]
        
        return cluster_keywords
    
    def get_cluster_sizes(self, labels: List[int]) -> Dict[int, int]:
        """
        Get size of each cluster
        """
        return dict(Counter(labels))
    
    def evaluate_clustering(self, labels: List[int]) -> float:
        """
        Evaluate clustering quality using silhouette score
        """
        if not self.is_fitted or self.tfidf_matrix is None:
            return 0.0
        
        n_labels = len(set(labels))
        if n_labels < 2 or n_labels >= len(labels):
            return 0.0
        
        try:
            score = silhouette_score(self.tfidf_matrix, labels)
            return float(score)
        except:
            return 0.0
    
    def find_optimal_clusters(self, documents: List[str], 
                             max_clusters: int = 10) -> int:
        """
        Find optimal number of clusters using elbow method
        """
        if len(documents) < 3:
            return 1
        
        cleaned_docs = [self._preprocess(doc) for doc in documents]
        tfidf_matrix = self.vectorizer.fit_transform(cleaned_docs)
        
        max_k = min(max_clusters, len(documents) - 1)
        if max_k < 2:
            return 1
        
        scores = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(tfidf_matrix)
            
            try:
                score = silhouette_score(tfidf_matrix, labels)
                scores.append((k, score))
            except:
                continue
        
        if not scores:
            return min(3, len(documents))
        
        best_k = max(scores, key=lambda x: x[1])[0]
        return best_k
    
    def _preprocess(self, text: str) -> str:
        """
        Preprocess text for clustering
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


def cluster_papers(papers: List[Dict], n_clusters: int = None) -> Dict:
    """
    Cluster papers and return cluster assignments with metadata
    """
    if not papers:
        return {'assignments': [], 'labels': {}, 'keywords': {}, 'sizes': {}, 'quality_score': 0, 'n_clusters': 0}
    
    texts = []
    for paper in papers:
        text = paper.get('title', '') + ' ' + paper.get('abstract', '')
        if not text.strip():
            text = paper.get('content', '')[:2000] if paper.get('content') else ''
        texts.append(text if text.strip() else 'empty document')
    
    valid_texts = [t for t in texts if t.strip() and len(t) > 10]
    if len(valid_texts) < 2:
        return {'assignments': [0] * len(papers), 'labels': {0: 'All Papers'}, 'keywords': {0: []}, 'sizes': {0: len(papers)}, 'quality_score': 0, 'n_clusters': 1}
    
    try:
        clusterer = PaperClusterer()
        
        if n_clusters is None:
            n_clusters = min(clusterer.find_optimal_clusters(texts), len(valid_texts))
        
        clusterer.n_clusters = max(1, n_clusters)
        labels = clusterer.fit_predict(texts)
        
        keywords = clusterer.get_cluster_keywords(texts, labels)
        sizes = clusterer.get_cluster_sizes(labels)
        quality = clusterer.evaluate_clustering(labels)
        
        cluster_labels = {}
        for cluster_id, kws in keywords.items():
            if kws:
                cluster_labels[cluster_id] = " / ".join(kws[:3]).title()
            else:
                cluster_labels[cluster_id] = f"Cluster {cluster_id + 1}"
        
        return {
            'assignments': labels,
            'keywords': keywords,
            'labels': cluster_labels,
            'sizes': sizes,
            'quality_score': quality,
            'n_clusters': len(set(labels))
        }
    except Exception as e:
        print(f"Clustering error: {e}")
        return {'assignments': [0] * len(papers), 'labels': {0: 'All Papers'}, 'keywords': {0: []}, 'sizes': {0: len(papers)}, 'quality_score': 0, 'n_clusters': 1}


def extract_topics(papers: List[Dict], n_topics: int = 5) -> Dict:
    """
    Extract topics from papers using LDA
    """
    if not papers:
        return {'topics': {}, 'document_topics': []}
    
    texts = []
    for paper in papers:
        text = paper.get('abstract', '') or paper.get('content', '')[:3000]
        texts.append(text)
    
    modeler = TopicModeler(n_topics=n_topics)
    modeler.fit(texts)
    
    if not modeler.is_fitted:
        return {'topics': {}, 'document_topics': []}
    
    topic_words = modeler.get_topic_words()
    topic_labels = modeler.get_topic_labels()
    doc_topics = modeler.get_document_topics(texts)
    
    return {
        'topics': topic_words,
        'labels': topic_labels,
        'document_topics': doc_topics
    }
