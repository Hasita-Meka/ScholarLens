// src/types/TS_papers.ts
export interface Author {
  id: string;
  name: string;
}

export interface Paper {
  id: string;
  title: string;
  abstract: string;
  publicationDate: string | Date;
  category: string;
  citationCount?: number;
  authors?: Author[];
  keywords?: string[];
  methods?: string[];
  pdfUrl: string;
}
