import React, { useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { useSearch } from '../context/SearchContext';
import SearchResult from '../components/SearchResult';
import { Loader2 } from 'lucide-react';

const ResultsPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { results, loading, error, performSearch } = useSearch();
  const query = searchParams.get('q');
  const page = parseInt(searchParams.get('page') || '1');
  // Placeholder: Assume 100 total results, 10 per page
  const totalResults = 100; // Replace with real value if available
  const perPage = 10;
  const totalPages = Math.ceil(totalResults / perPage);

  useEffect(() => {
    if (query) {
      performSearch(query, page);
    } else {
      navigate('/');
    }
  }, [query, page, performSearch, navigate]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="w-8 h-8 text-blue-600 animate-spin" />
        <span className="ml-2 text-gray-600">Searching...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h2 className="text-lg font-semibold text-red-800 mb-2">Error</h2>
          <p className="text-red-600">{error}</p>
          <button
            onClick={() => navigate('/')}
            className="mt-4 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
          >
            Return to Search
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full min-h-screen bg-slate-50 dark:bg-slate-900 px-2 md:px-0 py-8 flex flex-col items-center">
      <div className="w-full max-w-4xl mx-auto">
        {query && (
          <div className="mb-8 text-center">
            <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-2">
              Search Results for "{query}"
            </h1>
            <p className="text-gray-600 dark:text-gray-300 text-lg">
              Found {results.length} results
            </p>
          </div>
        )}

        {results.length === 0 ? (
          <div className="bg-white dark:bg-slate-800 rounded-lg shadow-md p-8 text-center">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
              No results found
            </h2>
            <p className="text-gray-600 dark:text-gray-300">
              We couldn't find any results for "{query}". Try using different keywords or broader terms.
            </p>
          </div>
        ) : (
          <div className="space-y-8">
            {results.map((result) => (
              <SearchResult key={result.url} result={result} />
            ))}
          </div>
        )}

        {/* Pagination */}
        {results.length > 0 && (
          <div className="mt-12 flex justify-center">
            <nav className="flex items-center space-x-2 bg-white dark:bg-slate-800 rounded-lg shadow border border-gray-200 dark:border-slate-700 px-4 py-2">
              <button
                onClick={() => navigate(`/search?q=${query}&page=${page - 1}`)}
                disabled={page === 1}
                className="px-3 py-1 rounded-md border border-gray-300 dark:border-slate-600 text-sm font-medium text-gray-700 dark:text-gray-200 bg-white dark:bg-slate-900 hover:bg-gray-50 dark:hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Previous
              </button>
              <span className="px-3 py-1 text-sm text-gray-700 dark:text-gray-200">
                Page {page} of {totalPages}
              </span>
              <button
                onClick={() => navigate(`/search?q=${query}&page=${page + 1}`)}
                disabled={page >= totalPages}
                className="px-3 py-1 rounded-md border border-gray-300 dark:border-slate-600 text-sm font-medium text-gray-700 dark:text-gray-200 bg-white dark:bg-slate-900 hover:bg-gray-50 dark:hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Next
              </button>
            </nav>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultsPage;