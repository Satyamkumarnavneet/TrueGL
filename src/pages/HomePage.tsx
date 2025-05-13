import React from 'react';
import SearchForm from '../components/SearchForm';
import { CheckCircle, Shield, Globe, AlertTriangle } from 'lucide-react';

const HomePage = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50 dark:bg-slate-900">
      <div className="w-full max-w-4xl px-4 py-8 text-center">
        <div className="mb-12">
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-blue-600 dark:text-blue-400 mb-4 tracking-tight">
            True<span className="text-slate-800 dark:text-white">GL</span>
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            Search the web with confidence. We analyze and score results based on factual accuracy.
          </p>
        </div>

        <SearchForm />

        <div className="mt-16 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 text-center">
          <div className="flex flex-col items-center p-4 rounded-lg bg-white dark:bg-slate-800 shadow-md hover:shadow-lg transition-shadow duration-200">
            <CheckCircle className="h-10 w-10 text-green-500 mb-3" />
            <h3 className="text-lg font-medium mb-2">Fact Checked</h3>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Results are analyzed against verified sources to assess accuracy.
            </p>
          </div>

          <div className="flex flex-col items-center p-4 rounded-lg bg-white dark:bg-slate-800 shadow-md hover:shadow-lg transition-shadow duration-200">
            <Shield className="h-10 w-10 text-blue-500 mb-3" />
            <h3 className="text-lg font-medium mb-2">Truth Score</h3>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Every result gets a truth score based on reliability and factual content.
            </p>
          </div>

          <div className="flex flex-col items-center p-4 rounded-lg bg-white dark:bg-slate-800 shadow-md hover:shadow-lg transition-shadow duration-200">
            <Globe className="h-10 w-10 text-teal-500 mb-3" />
            <h3 className="text-lg font-medium mb-2">Comprehensive</h3>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Search across the entire web while maintaining quality standards.
            </p>
          </div>

          <div className="flex flex-col items-center p-4 rounded-lg bg-white dark:bg-slate-800 shadow-md hover:shadow-lg transition-shadow duration-200">
            <AlertTriangle className="h-10 w-10 text-yellow-500 mb-3" />
            <h3 className="text-lg font-medium mb-2">Misinformation Alert</h3>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Clear warnings for content that contains potential misinformation.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;