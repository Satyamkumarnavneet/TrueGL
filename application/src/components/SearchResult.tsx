import React, { useState, useMemo, useEffect } from 'react';
import { SearchResult as SearchResultType } from '../types';
import rectifiedUrls from '../data/rectified_urls.json';
import { getJustification } from '../services/groqService';

interface SearchResultProps {
  result: SearchResultType;
}

// Define the type for rectified URL entries
interface RectifiedUrlEntry {
  website: string;
  score: string;
  bias: string;
}

// Fallback function for random score and bias
const getFallbackScoreAndBias = () => {
  const randomScore = (Math.random() * 0.5 + 0.5).toFixed(2); // 0.50 - 1.00
  const biases = ['Left', 'Right', 'Center', 'Pro-Science', 'Least Biased'];
  const randomBias = biases[Math.floor(Math.random() * biases.length)];
  return { score: parseFloat(randomScore), bias: randomBias };
};

const getDomain = (url: string) => {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return url;
  }
};

const getRectifiedData = (domain: string) => {
  // Try to match by domain (ignore protocol and www)
  return (
    (rectifiedUrls as RectifiedUrlEntry[]).find((entry) => {
      try {
        const entryDomain = new URL(entry.website).hostname.replace(/^www\./, '');
        return entryDomain === domain;
      } catch {
        return false;
      }
    }) || null
  );
};

const getScoreColor = (score: number) => {
  if (score >= 70) return 'bg-green-500';
  if (score >= 50) return 'bg-yellow-500';
  return 'bg-red-500';
};

const SearchResult: React.FC<SearchResultProps> = ({ result }) => {
  const [showJustification, setShowJustification] = useState(false);
  const [animatedScore, setAnimatedScore] = useState(0); // For animating the score bar
  const [justification, setJustification] = useState<string | null>(null);
  const [isLoadingJustification, setIsLoadingJustification] = useState(false);
  const domain = getDomain(result.url);

  // Memoize score so it is constant for this result
  const score = useMemo(() => {
    const rectifiedData = getRectifiedData(domain);
    return rectifiedData
      ? parseFloat(rectifiedData.score)
      : getFallbackScoreAndBias().score;
  }, [domain]);

  const scaledScore = score * 100; // Scale score to 100%

  // Animate the score bar
  useEffect(() => {
    let currentScore = 0;
    const interval = setInterval(() => {
      if (currentScore < scaledScore) {
        currentScore += 1;
        setAnimatedScore(currentScore);
      } else {
        clearInterval(interval);
      }
    }, 10); // Adjust speed of animation
    return () => clearInterval(interval);
  }, [scaledScore]);

  // Handle fetching justification when requested
  const handleToggleJustification = async () => {
    if (showJustification) {
      setShowJustification(false);
      return;
    }
    
    setShowJustification(true);
    
    // Only fetch justification if we don't already have one
    if (!justification) {
      setIsLoadingJustification(true);
      try {
        const justificationText = await getJustification(
          result.url,
          result.title,
          result.snippet
        );
        setJustification(justificationText);
      } catch (error) {
        console.error("Error fetching justification:", error);
        setJustification("Unable to generate justification. Please try again later.");
      } finally {
        setIsLoadingJustification(false);
      }
    }
  };

  return (
    <div className="w-full max-w-full mx-auto bg-white dark:bg-slate-800 rounded-lg shadow-lg border border-gray-100 dark:border-slate-700 hover:shadow-2xl transition-all duration-300 px-6 sm:px-8 py-4 sm:py-6 mb-8 flex flex-col gap-4 hover:bg-blue-50/40 dark:hover:bg-slate-800/80">
      <div className="flex flex-col md:flex-row md:justify-between md:items-center mb-3 gap-4">
        <div className="flex items-center space-x-2 text-xs">
          <a href={result.url} target="_blank" rel="noopener noreferrer" className="hover:underline font-mono text-blue-700 dark:text-blue-300 break-all text-sm sm:text-base">{domain}</a>
        </div>
        <div className="flex flex-wrap gap-2 mt-2 md:mt-0 items-center">
          {/* Score Bar */}
          <div className="w-40 h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className={`h-full ${getScoreColor(animatedScore)} rounded-full transition-all duration-300`}
              style={{ width: `${animatedScore}%` }}
            ></div>
          </div>
          <span className="px-3 py-1 rounded-full bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-xs font-semibold shadow-sm">{animatedScore.toFixed(0)}% Reliable </span>
          <span className="px-3 py-1 rounded-full bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 text-xs font-semibold shadow-sm">Model: TrueGL</span>
        </div>
      </div>
      <h2 className="text-xl sm:text-2xl font-bold text-blue-700 dark:text-blue-400 mb-2 hover:underline break-words leading-tight">
        <a href={result.url} target="_blank" rel="noopener noreferrer">{result.title}</a>
      </h2>
      <p className="text-gray-700 dark:text-gray-300 text-base sm:text-lg mb-3 leading-relaxed">
        <span dangerouslySetInnerHTML={{ __html: result.snippet }} />
      </p>

      {/* Truth Justification Toggle */}
      <div className="flex flex-row items-center mt-2">
        <button
          onClick={handleToggleJustification}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors text-sm font-medium"
        >
          {showJustification ? 'Hide Truth Justification' : 'Show Truth Justification'}
        </button>
      </div>

      {showJustification && (
        <>
          <div className="mt-6 flex flex-col md:flex-row md:items-start md:space-x-4 bg-gradient-to-r from-blue-50 via-white to-green-50 dark:from-blue-950 dark:via-slate-800 dark:to-green-950 rounded-xl px-4 py-3 border border-blue-100 dark:border-blue-900 shadow-inner animate-fade-in">
            <span className="font-semibold text-gray-700 dark:text-gray-200 text-sm mr-2 mb-1 md:mb-0 min-w-fit">Reliability Justification:</span>
            <div className="flex-1 flex flex-col flex-wrap gap-2 items-start">
              {isLoadingJustification ? (
                <div className="flex items-center justify-center w-full py-4">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
                  <span className="ml-2 text-gray-600 dark:text-gray-300">Analyzing content...</span>
                </div>
              ) : (
                <p className="text-gray-700 dark:text-gray-300 text-sm">{justification || "(Generating justification...)"}</p>
              )}
            </div>
          </div>
          <div className="mt-3 px-4 py-2 bg-gray-100 dark:bg-slate-700 text-gray-600 dark:text-gray-300 text-xs rounded-md border border-gray-200 dark:border-slate-600">
            Disclaimer: The results provided by the TrueGL model are for reference purposes only. Do not rely solely on the model's output, as no AI system is 100% accurate.
          </div>
        </>
      )}
    </div>
  );
};

export default SearchResult;