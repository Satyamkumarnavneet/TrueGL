import React from 'react';
import { SearchResult } from '../types';
import TruthScoreBadge from './TruthScoreBadge';
import { ExternalLink } from 'lucide-react';
import rectifiedUrls from '../data/rectified_urls.json'; // Import CSV data as JSON

interface SearchResultItemProps {
  result: SearchResult;
}

// Fallback function for random score and bias
const getFallbackScoreAndBias = () => {
  const randomScore = Math.floor(Math.random() * 50) + 50; // Random score between 50 and 100
  const biases = ['Left', 'Right', 'Center', 'Pro-Science', 'Least Biased'];
  const randomBias = biases[Math.floor(Math.random() * biases.length)];
  return { score: randomScore, bias: randomBias };
};

const SearchResultItem: React.FC<SearchResultItemProps> = ({ result }) => {
  const { title, url, snippet } = result;

  // Format the URL for display
  const displayUrl = new URL(url).hostname;

  // Get score and bias from CSV or fallback
  const rectifiedData = rectifiedUrls.find((entry) => entry.website === url);
  const { score, bias } = rectifiedData || getFallbackScoreAndBias();

  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg shadow-md p-6 mb-6 border border-gray-200 dark:border-slate-700 hover:shadow-lg transition-shadow duration-300">
      <div className="flex justify-between items-start mb-3">
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-gray-500 dark:text-gray-400 text-sm hover:underline flex items-center"
        >
          {displayUrl}
          <ExternalLink className="h-4 w-4 ml-2" />
        </a>
        <TruthScoreBadge score={score} bias={bias} />
      </div>

      <h2 className="text-xl font-semibold text-blue-700 dark:text-blue-400 hover:underline mb-3">
        <a href={url} target="_blank" rel="noopener noreferrer">
          {title}
        </a>
      </h2>

      <p className="text-gray-700 dark:text-gray-300 text-sm leading-relaxed mb-4">
        {snippet}
      </p>

      <div className="flex justify-between items-center text-xs text-gray-500 dark:text-gray-400">
        <div className="flex space-x-4">
          <button className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
            Similar results
          </button>
          <button className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
            Share
          </button>
        </div>
        <span>
          {new Date(result.dateIndexed).toLocaleDateString(undefined, {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
          })}
        </span>
      </div>
    </div>
  );
};

export default SearchResultItem;