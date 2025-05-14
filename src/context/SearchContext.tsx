import React, { createContext, useContext, useState, useCallback, useMemo } from 'react';
import { SearchContextType, SearchResult, SearchFilters } from '../types';
import rectifiedUrls from '../data/rectified_urls.json';

const SearchContext = createContext<SearchContextType | undefined>(undefined);

export const useSearch = () => {
  const context = useContext(SearchContext);
  if (!context) {
    throw new Error('useSearch must be used within a SearchProvider');
  }
  return context;
};

interface SearchProviderProps {
  children: React.ReactNode;
}

// Helper to get bias for a result
const getBias = (result: SearchResult): string => {
  try {
    const domain = new URL(result.url).hostname.replace(/^www\./, '');
    const entry = (rectifiedUrls as any[]).find((e) => {
      try {
        const entryDomain = new URL(e.website).hostname.replace(/^www\./, '');
        return entryDomain === domain;
      } catch {
        return false;
      }
    });
    return entry ? entry.bias : 'Unknown';
  } catch {
    return 'Unknown';
  }
};

export const SearchProvider: React.FC<SearchProviderProps> = ({ children }) => {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<SearchFilters>({});
  const [sort, setSort] = useState<string>('');

  const performSearch = useCallback(async (query: string, offset: number = 0) => {
    setLoading(true);
    setError(null);

    try {
      console.log(`SearchContext: Performing search with query="${query}", offset=${offset}`);
      
      const response = await fetch('http://localhost:8000/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({
          query,
          offset,
          per_page: 10,
          filters,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        // Handle structured error response
        const errorMessage = data.detail?.message || data.detail?.error || response.statusText;
        throw new Error(`Search failed: ${errorMessage}`);
      }

      if (!data.results) {
        throw new Error('Invalid response format from server');
      }

      console.log(`SearchContext: Search successful, received ${data.results.length} results for offset ${offset}`);
      setResults(data.results);
    } catch (err) {
      console.error('Search error:', err);
      const errorMessage = err instanceof Error ? err.message : 'An error occurred during search';
      setError(errorMessage);
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, [filters]);

  const clearResults = useCallback(() => {
    setResults([]);
    setError(null);
  }, []);

  const updateFilters = useCallback((newFilters: Partial<SearchFilters>) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
  }, []);

  // Sorting logic
  const sortedResults = useMemo(() => {
    if (!sort) return results;
    const sorted = [...results];
    switch (sort) {
      case 'score_desc':
        sorted.sort((a, b) => b.score - a.score);
        break;
      case 'score_asc':
        sorted.sort((a, b) => a.score - b.score);
        break;
      case 'bias_az':
        sorted.sort((a, b) => getBias(a).localeCompare(getBias(b)));
        break;
      case 'bias_za':
        sorted.sort((a, b) => getBias(b).localeCompare(getBias(a)));
        break;
      case 'title_az':
        sorted.sort((a, b) => a.title.localeCompare(b.title));
        break;
      case 'title_za':
        sorted.sort((a, b) => b.title.localeCompare(a.title));
        break;
      default:
        break;
    }
    return sorted;
  }, [results, sort]);

  const value: SearchContextType = {
    results: sortedResults,
    loading,
    error,
    performSearch,
    clearResults,
    filters,
    updateFilters,
    sort,
    setSort,
  };

  return (
    <SearchContext.Provider value={value}>
      {children}
    </SearchContext.Provider>
  );
};

export default SearchContext;