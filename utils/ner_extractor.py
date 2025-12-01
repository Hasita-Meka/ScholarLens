"""
Named Entity Recognition for ScholarLens
Extracts methods, datasets, authors, and institutions from research papers
"""

import re
from typing import List, Dict, Set, Tuple


KNOWN_METHODS = {
    'neural network', 'deep learning', 'machine learning', 'cnn', 'rnn', 'lstm',
    'transformer', 'bert', 'gpt', 'attention mechanism', 'self-attention',
    'convolutional neural network', 'recurrent neural network', 'gan',
    'generative adversarial network', 'autoencoder', 'variational autoencoder',
    'reinforcement learning', 'supervised learning', 'unsupervised learning',
    'transfer learning', 'fine-tuning', 'pre-training', 'embedding',
    'word2vec', 'glove', 'fasttext', 'elmo', 'xlnet', 'roberta', 'albert',
    't5', 'bart', 'gpt-2', 'gpt-3', 'gpt-4', 'llama', 'palm', 'claude',
    'diffusion model', 'stable diffusion', 'dalle', 'midjourney',
    'random forest', 'decision tree', 'svm', 'support vector machine',
    'logistic regression', 'linear regression', 'naive bayes',
    'k-means', 'clustering', 'pca', 'principal component analysis',
    'gradient descent', 'sgd', 'adam', 'optimizer', 'backpropagation',
    'dropout', 'batch normalization', 'layer normalization',
    'resnet', 'vgg', 'inception', 'efficientnet', 'mobilenet',
    'yolo', 'faster rcnn', 'mask rcnn', 'unet', 'segmentation',
    'object detection', 'image classification', 'semantic segmentation',
    'named entity recognition', 'ner', 'pos tagging', 'parsing',
    'sentiment analysis', 'text classification', 'question answering',
    'machine translation', 'summarization', 'text generation',
    'knowledge graph', 'graph neural network', 'gnn', 'gcn',
    'contrastive learning', 'self-supervised learning', 'few-shot learning',
    'zero-shot learning', 'meta-learning', 'multi-task learning',
    'cross-entropy loss', 'mse loss', 'focal loss', 'triplet loss',
    'beam search', 'greedy decoding', 'nucleus sampling', 'top-k sampling',
    'rag', 'retrieval augmented generation', 'chain of thought', 'cot',
    'prompt engineering', 'in-context learning', 'instruction tuning',
    'rlhf', 'reinforcement learning from human feedback', 'dpo',
}

KNOWN_DATASETS = {
    'imagenet', 'cifar-10', 'cifar-100', 'mnist', 'fashion-mnist',
    'coco', 'pascal voc', 'cityscapes', 'ade20k', 'lvis',
    'squad', 'glue', 'superglue', 'mrpc', 'sst-2', 'mnli', 'qnli',
    'wikitext', 'penn treebank', 'ptb', 'billion word', 'bookcorpus',
    'wikipedia', 'common crawl', 'c4', 'pile', 'redpajama',
    'imdb', 'yelp', 'amazon reviews', 'sentiment140',
    'conll-2003', 'ontonotes', 'ace2005',
    'wmt', 'iwslt', 'europarl', 'opus',
    'ms marco', 'natural questions', 'triviaqa', 'hotpotqa',
    'pubmed', 'arxiv', 'semantic scholar',
    'kinetics', 'ucf101', 'hmdb51', 'youtube-8m',
    'librispeech', 'voxceleb', 'commonvoice',
    'laion', 'conceptual captions', 'cc3m', 'cc12m',
    'webtext', 'openwebtext', 'refinedweb',
    'humaneval', 'mbpp', 'apps', 'code contest',
    'mmlu', 'hellaswag', 'winogrande', 'arc',
    'gsm8k', 'math', 'big-bench',
}

INSTITUTION_PATTERNS = [
    r'University of [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
    r'[A-Z][a-z]+\s+University',
    r'[A-Z][a-z]+\s+Institute of Technology',
    r'MIT|Stanford|Harvard|Oxford|Cambridge|Berkeley|CMU|Caltech',
    r'Google(?:\s+Research)?|Microsoft(?:\s+Research)?|Meta(?:\s+AI)?|OpenAI|DeepMind|Anthropic',
    r'IBM(?:\s+Research)?|Amazon(?:\s+AI)?|Apple(?:\s+ML)?|NVIDIA(?:\s+Research)?',
    r'[A-Z][a-z]+\s+Labs?',
    r'[A-Z][a-z]+\s+Research(?:\s+Center)?',
]


def extract_entities(text: str) -> Dict[str, List[Dict]]:
    """
    Extract all entity types from text
    """
    return {
        'methods': extract_methods(text),
        'datasets': extract_datasets(text),
        'authors': extract_authors(text),
        'institutions': extract_institutions(text),
    }


def extract_methods(text: str) -> List[Dict]:
    """
    Extract research methods and techniques from text
    """
    text_lower = text.lower()
    found_methods = []
    seen = set()
    
    for method in KNOWN_METHODS:
        if method in text_lower and method not in seen:
            count = text_lower.count(method)
            context = find_context(text, method)
            found_methods.append({
                'name': method.title(),
                'raw_name': method,
                'count': count,
                'context': context,
                'category': categorize_method(method)
            })
            seen.add(method)
    
    abbreviation_patterns = [
        r'\b([A-Z]{2,6})\s*\(',
        r'\(\s*([A-Z]{2,6})\s*\)',
    ]
    
    for pattern in abbreviation_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match) >= 2 and match.lower() not in seen:
                if is_likely_method_abbrev(match):
                    found_methods.append({
                        'name': match,
                        'raw_name': match.lower(),
                        'count': text.count(match),
                        'context': find_context(text, match),
                        'category': 'unknown'
                    })
                    seen.add(match.lower())
    
    return sorted(found_methods, key=lambda x: x['count'], reverse=True)


def extract_datasets(text: str) -> List[Dict]:
    """
    Extract datasets from text
    """
    text_lower = text.lower()
    found_datasets = []
    seen = set()
    
    for dataset in KNOWN_DATASETS:
        if dataset in text_lower and dataset not in seen:
            count = text_lower.count(dataset)
            context = find_context(text, dataset)
            found_datasets.append({
                'name': dataset.upper() if len(dataset) <= 6 else dataset.title(),
                'raw_name': dataset,
                'count': count,
                'context': context,
                'domain': categorize_dataset(dataset)
            })
            seen.add(dataset)
    
    dataset_patterns = [
        r'([A-Z][a-z]*(?:-[A-Z]?[a-z]+)*)\s+dataset',
        r'dataset\s+called\s+([A-Z][a-z]+(?:-[a-z]+)*)',
        r'benchmark(?:ed)?\s+on\s+([A-Z][a-z]+(?:-[0-9]+)?)',
    ]
    
    for pattern in dataset_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if match.lower() not in seen and len(match) > 2:
                found_datasets.append({
                    'name': match,
                    'raw_name': match.lower(),
                    'count': 1,
                    'context': find_context(text, match),
                    'domain': 'unknown'
                })
                seen.add(match.lower())
    
    return sorted(found_datasets, key=lambda x: x['count'], reverse=True)


def extract_authors(text: str) -> List[Dict]:
    """
    Extract author names from text
    """
    authors = []
    seen = set()
    
    header_text = text[:3000]
    
    name_patterns = [
        r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)',
        r'([A-Z]\.\s*[A-Z][a-z]+)',
        r'([A-Z][a-z]+,\s*[A-Z]\.)',
    ]
    
    for pattern in name_patterns:
        matches = re.findall(pattern, header_text)
        for match in matches:
            name = match.strip()
            name_key = name.lower().replace('.', '').replace(',', '')
            
            if name_key not in seen and is_likely_author_name(name):
                authors.append({
                    'name': name,
                    'normalized_name': normalize_author_name(name)
                })
                seen.add(name_key)
    
    return authors[:20]


def extract_institutions(text: str) -> List[Dict]:
    """
    Extract institutions from text
    """
    institutions = []
    seen = set()
    
    header_text = text[:5000]
    
    for pattern in INSTITUTION_PATTERNS:
        matches = re.findall(pattern, header_text)
        for match in matches:
            if match.lower() not in seen:
                institutions.append({
                    'name': match,
                    'type': categorize_institution(match)
                })
                seen.add(match.lower())
    
    return institutions


def find_context(text: str, term: str, window: int = 100) -> str:
    """
    Find surrounding context for a term
    """
    text_lower = text.lower()
    term_lower = term.lower()
    
    idx = text_lower.find(term_lower)
    if idx == -1:
        return ""
    
    start = max(0, idx - window)
    end = min(len(text), idx + len(term) + window)
    
    context = text[start:end].strip()
    context = re.sub(r'\s+', ' ', context)
    
    return context


def categorize_method(method: str) -> str:
    """
    Categorize a method into a domain
    """
    method_lower = method.lower()
    
    nlp_keywords = ['bert', 'gpt', 'transformer', 'attention', 'language', 'text', 'nlp', 'word', 'embedding', 'translation', 'summarization', 'ner', 'pos', 'parsing']
    cv_keywords = ['cnn', 'image', 'vision', 'detection', 'segmentation', 'resnet', 'vgg', 'yolo', 'rcnn', 'unet']
    dl_keywords = ['neural', 'deep', 'learning', 'network', 'layer', 'backpropagation', 'gradient', 'optimizer']
    gen_keywords = ['generative', 'gan', 'diffusion', 'autoencoder', 'vae', 'dalle', 'stable']
    
    if any(kw in method_lower for kw in nlp_keywords):
        return 'nlp'
    elif any(kw in method_lower for kw in cv_keywords):
        return 'computer_vision'
    elif any(kw in method_lower for kw in gen_keywords):
        return 'generative_ai'
    elif any(kw in method_lower for kw in dl_keywords):
        return 'deep_learning'
    else:
        return 'machine_learning'


def categorize_dataset(dataset: str) -> str:
    """
    Categorize a dataset into a domain
    """
    dataset_lower = dataset.lower()
    
    if any(kw in dataset_lower for kw in ['imagenet', 'cifar', 'mnist', 'coco', 'voc', 'cityscapes', 'ade', 'lvis', 'laion']):
        return 'computer_vision'
    elif any(kw in dataset_lower for kw in ['squad', 'glue', 'wiki', 'book', 'web', 'crawl', 'pile', 'imdb', 'conll', 'wmt']):
        return 'nlp'
    elif any(kw in dataset_lower for kw in ['kinetics', 'ucf', 'hmdb', 'youtube']):
        return 'video'
    elif any(kw in dataset_lower for kw in ['libri', 'vox', 'voice']):
        return 'audio'
    elif any(kw in dataset_lower for kw in ['humaneval', 'mbpp', 'code']):
        return 'code'
    else:
        return 'general'


def categorize_institution(name: str) -> str:
    """
    Categorize institution type
    """
    name_lower = name.lower()
    
    if 'university' in name_lower or 'institute' in name_lower or 'college' in name_lower:
        return 'academic'
    elif any(company in name_lower for company in ['google', 'microsoft', 'meta', 'amazon', 'apple', 'nvidia', 'ibm']):
        return 'industry'
    elif 'research' in name_lower or 'lab' in name_lower:
        return 'research_lab'
    else:
        return 'other'


def is_likely_method_abbrev(abbrev: str) -> bool:
    """
    Check if an abbreviation is likely a method name
    """
    excluded = {'THE', 'AND', 'FOR', 'WITH', 'FROM', 'THIS', 'THAT', 'HAVE', 'BEEN', 'ALSO', 'MORE', 'WERE', 'OUR', 'ARE', 'BUT', 'NOT', 'CAN', 'ALL', 'WAS', 'HAS'}
    return abbrev not in excluded and len(abbrev) >= 2


def is_likely_author_name(name: str) -> bool:
    """
    Check if a string is likely an author name
    """
    excluded_words = {
        'the', 'and', 'for', 'with', 'from', 'this', 'that', 'which', 'where', 'when',
        'abstract', 'introduction', 'method', 'result', 'conclusion', 'figure', 'table', 
        'section', 'chapter', 'appendix', 'reference', 'acknowledgment',
        'model', 'network', 'learning', 'neural', 'deep', 'machine', 'algorithm',
        'encoding', 'decoding', 'training', 'testing', 'validation', 'evaluation',
        'function', 'loss', 'optimization', 'gradient', 'layer', 'weight', 'bias',
        'attention', 'transformer', 'embedding', 'vector', 'matrix', 'tensor',
        'linear', 'nonlinear', 'convolutional', 'recurrent', 'generative', 'discriminative',
        'classification', 'regression', 'segmentation', 'detection', 'recognition',
        'prediction', 'generation', 'translation', 'summarization', 'extraction',
        'distance', 'signed', 'field', 'control', 'avoidance', 'collision', 'trajectory',
        'policy', 'reward', 'state', 'action', 'agent', 'environment', 'simulation',
        'robot', 'aerial', 'robots', 'autonomous', 'navigation', 'planning', 'path',
        'sensor', 'camera', 'lidar', 'imu', 'encoder', 'decoder', 'feature', 'features',
        'input', 'output', 'hidden', 'latent', 'representation', 'space', 'dimension',
        'batch', 'epoch', 'iteration', 'step', 'rate', 'schedule', 'warmup', 'decay',
        'accuracy', 'precision', 'recall', 'score', 'metric', 'error', 'loss',
        'positional', 'relative', 'absolute', 'global', 'local', 'spatial', 'temporal',
        'multi', 'single', 'dual', 'triple', 'cross', 'self', 'auto', 'semi', 'fully',
        'based', 'driven', 'aware', 'guided', 'constrained', 'regularized', 'normalized',
        'reinforcement', 'supervised', 'unsupervised', 'transfer', 'contrastive', 'federated',
        'title', 'journal', 'proceedings', 'conference', 'workshop', 'arxiv', 'preprint',
        'digital', 'substation', 'substations', 'virtualized', 'protection', 'virtual',
        'predictive', 'adaptive', 'dynamic', 'static', 'hybrid', 'integrated', 'distributed',
        'analysis', 'system', 'systems', 'framework', 'architecture', 'design', 'implementation',
        'approach', 'technique', 'techniques', 'strategies', 'strategy', 'solution', 'solutions',
        'data', 'dataset', 'datasets', 'benchmark', 'benchmarks', 'evaluation', 'evaluations',
        'power', 'energy', 'electric', 'electrical', 'voltage', 'current', 'frequency',
        'grid', 'smart', 'cyber', 'security', 'attack', 'defense', 'threat', 'vulnerability',
        'communication', 'protocol', 'protocols', 'standard', 'standards', 'specification',
        'real', 'time', 'online', 'offline', 'continuous', 'discrete', 'optimal', 'optimized',
    }
    
    technical_patterns = [
        r'.*\s+(model|network|learning|algorithm|method|system|framework|architecture)$',
        r'.*\s+(encoding|decoding|processing|training|inference)$',
        r'.*\s+(function|layer|module|block|unit|cell)$',
        r'.*\s+(control|planning|navigation|avoidance|detection)$',
        r'.*\s+(distance|field|space|representation|embedding)$',
        r'^(neural|deep|machine|reinforcement|supervised|unsupervised)\s+.*',
        r'^(linear|nonlinear|convolutional|recurrent|attention)\s+.*',
        r'^(signed|positional|relative|absolute|global|local)\s+.*',
        r'^(transfer|contrastive|federated|meta|few-shot|zero-shot)\s+.*',
        r'^title\s+.*',
    ]
    
    name_lower = name.lower()
    name_normalized = re.sub(r'[^a-z\s]', '', name_lower).strip()
    
    if name_normalized in KNOWN_METHODS:
        return False
    
    for known_method in KNOWN_METHODS:
        if name_normalized == known_method or known_method == name_normalized:
            return False
        if ' ' in known_method:
            method_words = set(known_method.split())
            name_words = set(name_normalized.split())
            if method_words == name_words:
                return False
    
    for pattern in technical_patterns:
        if re.match(pattern, name_lower):
            return False
    
    words = name_lower.split()
    if any(word in excluded_words for word in words):
        return False
    
    if len(name) < 3 or len(name) > 50:
        return False
    
    if len(words) < 2:
        return False
    
    if all(len(word) <= 2 for word in words):
        return False
    
    return True


def normalize_author_name(name: str) -> str:
    """
    Normalize author name format
    """
    name = name.strip()
    name = re.sub(r'\s+', ' ', name)
    
    if ',' in name:
        parts = name.split(',')
        if len(parts) == 2:
            name = f"{parts[1].strip()} {parts[0].strip()}"
    
    return name


def build_method_prerequisites() -> Dict[str, List[str]]:
    """
    Build a mapping of methods to their prerequisites
    """
    return {
        'Transformer': ['Attention Mechanism', 'Neural Network'],
        'Bert': ['Transformer', 'Pre-Training'],
        'Gpt': ['Transformer', 'Language Model'],
        'Gpt-2': ['Gpt', 'Transformer'],
        'Gpt-3': ['Gpt-2', 'Transformer'],
        'Gpt-4': ['Gpt-3', 'Transformer'],
        'Attention Mechanism': ['Neural Network', 'Rnn'],
        'Self-Attention': ['Attention Mechanism'],
        'Lstm': ['Rnn', 'Neural Network'],
        'Rnn': ['Neural Network', 'Backpropagation'],
        'Cnn': ['Neural Network', 'Convolution'],
        'Resnet': ['Cnn', 'Batch Normalization'],
        'Gan': ['Neural Network', 'Generative Model'],
        'Diffusion Model': ['Generative Model', 'Neural Network'],
        'Stable Diffusion': ['Diffusion Model', 'Autoencoder'],
        'Variational Autoencoder': ['Autoencoder', 'Bayesian'],
        'Reinforcement Learning': ['Machine Learning', 'Markov Decision Process'],
        'Transfer Learning': ['Deep Learning', 'Pre-Training'],
        'Fine-Tuning': ['Transfer Learning', 'Pre-Training'],
        'Rag': ['Transformer', 'Information Retrieval'],
        'Chain Of Thought': ['Prompt Engineering', 'Language Model'],
        'Rlhf': ['Reinforcement Learning', 'Language Model'],
    }
