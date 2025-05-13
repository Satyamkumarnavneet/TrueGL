export interface SearchResult {
  url: string;
  title: string;
  snippet: string;
  score: number;
  domain: string;
  language: string;
  metadata: {
    published_date?: string;
    age?: string;
    type?: string;
  };
  last_updated: string;
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
}

export interface SearchContextType {
  results: SearchResult[];
  loading: boolean;
  error: string | null;
  performSearch: (query: string, page?: number) => Promise<void>;
  clearResults: () => void;
  filters: SearchFilters;
  updateFilters: (filters: Partial<SearchFilters>) => void;
  sort: string;
  setSort: (sort: string) => void;
}

export interface SearchFilters {
  min_date?: string;
  max_date?: string;
  language?: string[];
  sort?: string;
} 