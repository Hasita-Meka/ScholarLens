"""
arXiv and PubMed API integration for ScholarLens
Fetches research papers from open-access sources
"""

import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from datetime import datetime
import re
import time


class ArxivAPI:
    """
    arXiv API client for fetching research papers
    """
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self):
        self.session = requests.Session()
    
    def search(self, query: str, max_results: int = 10, 
               category: str = None, sort_by: str = "relevance") -> List[Dict]:
        """
        Search arXiv for papers matching the query
        
        Args:
            query: Search query string
            max_results: Maximum number of results (default 10, max 100)
            category: arXiv category filter (e.g., 'cs.AI', 'cs.LG')
            sort_by: Sort order ('relevance', 'lastUpdatedDate', 'submittedDate')
        """
        search_query = f"all:{query}"
        if category:
            search_query = f"cat:{category} AND {search_query}"
        
        sort_mapping = {
            'relevance': 'relevance',
            'lastUpdatedDate': 'lastUpdatedDate',
            'submittedDate': 'submittedDate'
        }
        
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': min(max_results, 100),
            'sortBy': sort_mapping.get(sort_by, 'relevance'),
            'sortOrder': 'descending'
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            return self._parse_response(response.text)
        except Exception as e:
            print(f"arXiv API error: {e}")
            return []
    
    def get_paper_by_id(self, arxiv_id: str) -> Optional[Dict]:
        """
        Get a specific paper by its arXiv ID
        """
        arxiv_id = arxiv_id.replace('arXiv:', '').strip()
        
        params = {
            'id_list': arxiv_id,
            'max_results': 1
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            papers = self._parse_response(response.text)
            return papers[0] if papers else None
        except Exception as e:
            print(f"arXiv API error: {e}")
            return None
    
    def _parse_response(self, xml_content: str) -> List[Dict]:
        """
        Parse arXiv API XML response
        """
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            for entry in root.findall('atom:entry', ns):
                paper = {}
                
                title_elem = entry.find('atom:title', ns)
                paper['title'] = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else ''
                
                abstract_elem = entry.find('atom:summary', ns)
                paper['abstract'] = abstract_elem.text.strip().replace('\n', ' ') if abstract_elem is not None else ''
                
                paper['authors'] = []
                for author in entry.findall('atom:author', ns):
                    name_elem = author.find('atom:name', ns)
                    if name_elem is not None:
                        paper['authors'].append({'name': name_elem.text})
                
                id_elem = entry.find('atom:id', ns)
                if id_elem is not None:
                    paper['arxiv_id'] = id_elem.text.split('/abs/')[-1]
                    paper['url'] = id_elem.text
                
                published_elem = entry.find('atom:published', ns)
                if published_elem is not None:
                    try:
                        pub_date = datetime.fromisoformat(published_elem.text.replace('Z', '+00:00'))
                        paper['publication_date'] = pub_date
                        paper['year'] = pub_date.year
                    except:
                        paper['year'] = None
                
                paper['categories'] = []
                for cat in entry.findall('atom:category', ns):
                    term = cat.get('term')
                    if term:
                        paper['categories'].append(term)
                
                for link in entry.findall('atom:link', ns):
                    if link.get('title') == 'pdf':
                        paper['pdf_url'] = link.get('href')
                        break
                
                doi_elem = entry.find('arxiv:doi', ns)
                if doi_elem is not None:
                    paper['doi'] = doi_elem.text
                
                paper['source'] = 'arxiv'
                papers.append(paper)
            
        except ET.ParseError as e:
            print(f"XML parse error: {e}")
        
        return papers
    
    def get_categories(self) -> Dict[str, str]:
        """
        Return common arXiv categories
        """
        return {
            'cs.AI': 'Artificial Intelligence',
            'cs.LG': 'Machine Learning',
            'cs.CL': 'Computation and Language (NLP)',
            'cs.CV': 'Computer Vision',
            'cs.NE': 'Neural and Evolutionary Computing',
            'cs.IR': 'Information Retrieval',
            'cs.RO': 'Robotics',
            'stat.ML': 'Machine Learning (Statistics)',
            'math.OC': 'Optimization and Control',
            'eess.SP': 'Signal Processing',
        }


class PubMedAPI:
    """
    PubMed API client for fetching biomedical research papers
    """
    
    SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.session = requests.Session()
    
    def search(self, query: str, max_results: int = 10,
               min_date: str = None, max_date: str = None) -> List[Dict]:
        """
        Search PubMed for papers matching the query
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            min_date: Minimum publication date (YYYY/MM/DD)
            max_date: Maximum publication date (YYYY/MM/DD)
        """
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': min(max_results, 100),
            'retmode': 'json',
            'sort': 'relevance'
        }
        
        if self.api_key:
            search_params['api_key'] = self.api_key
        
        if min_date:
            search_params['mindate'] = min_date
        if max_date:
            search_params['maxdate'] = max_date
        
        try:
            response = self.session.get(self.SEARCH_URL, params=search_params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            id_list = data.get('esearchresult', {}).get('idlist', [])
            
            if not id_list:
                return []
            
            time.sleep(0.34)
            
            return self._fetch_papers(id_list)
        except Exception as e:
            print(f"PubMed search error: {e}")
            return []
    
    def _fetch_papers(self, pmids: List[str]) -> List[Dict]:
        """
        Fetch full paper details for given PubMed IDs
        """
        fetch_params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml',
            'rettype': 'abstract'
        }
        
        if self.api_key:
            fetch_params['api_key'] = self.api_key
        
        try:
            response = self.session.get(self.FETCH_URL, params=fetch_params, timeout=30)
            response.raise_for_status()
            
            return self._parse_pubmed_xml(response.text)
        except Exception as e:
            print(f"PubMed fetch error: {e}")
            return []
    
    def _parse_pubmed_xml(self, xml_content: str) -> List[Dict]:
        """
        Parse PubMed XML response
        """
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for article in root.findall('.//PubmedArticle'):
                paper = {}
                
                medline = article.find('.//MedlineCitation')
                if medline is None:
                    continue
                
                pmid_elem = medline.find('.//PMID')
                paper['pmid'] = pmid_elem.text if pmid_elem is not None else ''
                
                article_elem = medline.find('.//Article')
                if article_elem is None:
                    continue
                
                title_elem = article_elem.find('.//ArticleTitle')
                paper['title'] = title_elem.text if title_elem is not None else ''
                
                abstract_elem = article_elem.find('.//Abstract/AbstractText')
                if abstract_elem is not None:
                    paper['abstract'] = abstract_elem.text or ''
                else:
                    paper['abstract'] = ''
                
                paper['authors'] = []
                for author in article_elem.findall('.//Author'):
                    last_name = author.find('LastName')
                    first_name = author.find('ForeName')
                    if last_name is not None:
                        name = last_name.text
                        if first_name is not None:
                            name = f"{first_name.text} {name}"
                        paper['authors'].append({'name': name})
                
                pub_date = article_elem.find('.//PubDate')
                if pub_date is not None:
                    year_elem = pub_date.find('Year')
                    if year_elem is not None:
                        try:
                            paper['year'] = int(year_elem.text)
                        except:
                            paper['year'] = None
                
                journal_elem = article_elem.find('.//Journal/Title')
                paper['journal'] = journal_elem.text if journal_elem is not None else ''
                
                doi_elem = article.find('.//ArticleId[@IdType="doi"]')
                if doi_elem is not None:
                    paper['doi'] = doi_elem.text
                
                paper['url'] = f"https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/"
                paper['source'] = 'pubmed'
                
                mesh_terms = []
                for mesh in medline.findall('.//MeshHeading/DescriptorName'):
                    if mesh.text:
                        mesh_terms.append(mesh.text)
                paper['mesh_terms'] = mesh_terms
                
                papers.append(paper)
            
        except ET.ParseError as e:
            print(f"XML parse error: {e}")
        
        return papers
    
    def get_paper_by_pmid(self, pmid: str) -> Optional[Dict]:
        """
        Get a specific paper by its PubMed ID
        """
        papers = self._fetch_papers([pmid])
        return papers[0] if papers else None


def download_arxiv_pdf(arxiv_id: str, save_path: str) -> bool:
    """
    Download PDF from arXiv
    """
    arxiv_id = arxiv_id.replace('arXiv:', '').strip()
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    try:
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        return True
    except Exception as e:
        print(f"PDF download error: {e}")
        return False


def search_papers(query: str, sources: List[str] = None, 
                  max_results: int = 10) -> List[Dict]:
    """
    Search across multiple paper sources with deduplication.
    When multiple sources are used, max_results is the TOTAL limit.
    """
    if sources is None:
        sources = ['arxiv', 'pubmed']
    
    num_sources = len(sources)
    per_source_limit = max(1, max_results // num_sources) if num_sources > 1 else max_results
    
    all_papers = []
    seen_titles = set()
    
    if 'arxiv' in sources:
        try:
            arxiv = ArxivAPI()
            arxiv_papers = arxiv.search(query, max_results=per_source_limit)
            for paper in arxiv_papers:
                title_key = paper.get('title', '').lower().strip()[:100]
                if title_key and title_key not in seen_titles:
                    seen_titles.add(title_key)
                    all_papers.append(paper)
        except Exception as e:
            print(f"arXiv search error: {e}")
    
    if 'pubmed' in sources:
        try:
            pubmed = PubMedAPI()
            pubmed_papers = pubmed.search(query, max_results=per_source_limit)
            for paper in pubmed_papers:
                title_key = paper.get('title', '').lower().strip()[:100]
                if title_key and title_key not in seen_titles:
                    seen_titles.add(title_key)
                    all_papers.append(paper)
        except Exception as e:
            print(f"PubMed search error: {e}")
    
    return all_papers[:max_results]
