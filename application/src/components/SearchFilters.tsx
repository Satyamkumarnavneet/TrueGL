import React, { useState } from 'react';
import { useSearch } from '../context/SearchContext';

const sortOptions = [
  { value: '', label: 'Sort By' },
  { value: 'score_desc', label: 'Score: High to Low' },
  { value: 'score_asc', label: 'Score: Low to High' },
  { value: 'bias_az', label: 'Bias: A-Z' },
  { value: 'bias_za', label: 'Bias: Z-A' },
  { value: 'title_az', label: 'Title: A-Z' },
  { value: 'title_za', label: 'Title: Z-A' },
];

const SearchFilters: React.FC = () => {
  const { filters, updateFilters } = useSearch();
  const [sort, setSort] = useState(filters.sort || '');

  const handleSortChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    setSort(value);
    updateFilters({ sort: value });
  };

  const handleDateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    updateFilters({ [name]: value || undefined });
  };

  const handleLanguageChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    updateFilters({ language: value ? [value] : undefined });
  };

  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg shadow-md p-6 border border-gray-200 dark:border-slate-700">
      <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
        Filter Results
      </h2>

      {/* Date Range Filter */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Date Range
        </label>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
              From
            </label>
            <input
              type="date"
              name="min_date"
              value={filters.min_date || ''}
              onChange={handleDateChange}
              className="w-full rounded-md border border-gray-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-gray-900 dark:text-gray-100 shadow-sm focus:ring-blue-500 focus:border-blue-500 px-2 py-1"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
              To
            </label>
            <input
              type="date"
              name="max_date"
              value={filters.max_date || ''}
              onChange={handleDateChange}
              className="w-full rounded-md border border-gray-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-gray-900 dark:text-gray-100 shadow-sm focus:ring-blue-500 focus:border-blue-500 px-2 py-1"
            />
          </div>
        </div>
      </div>

      {/* Language Filter */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Language
        </label>
        <select
          value={filters.language?.[0] || ''}
          onChange={handleLanguageChange}
          className="w-full rounded-md border border-gray-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-gray-900 dark:text-gray-100 shadow-sm focus:ring-blue-500 focus:border-blue-500 px-2 py-1"
        >
          <option value="">All Languages</option>
          <option value="en">English</option>
          <option value="es">Spanish</option>
          <option value="fr">French</option>
          <option value="de">German</option>
          <option value="it">Italian</option>
          <option value="pt">Portuguese</option>
          <option value="ru">Russian</option>
          <option value="zh">Chinese</option>
          <option value="ja">Japanese</option>
          <option value="ko">Korean</option>
        </select>
      </div>

      {/* Sort Dropdown */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Sort Results
        </label>
        <select
          value={sort}
          onChange={handleSortChange}
          className="w-full rounded-md border border-gray-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-gray-900 dark:text-gray-100 shadow-sm focus:ring-blue-500 focus:border-blue-500 px-2 py-1"
        >
          {sortOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      {/* Clear Filters Button */}
      <button
        onClick={() => updateFilters({})}
        className="w-full py-2 px-4 bg-gray-100 dark:bg-slate-700 text-gray-700 dark:text-gray-200 rounded-md shadow-sm hover:bg-gray-200 dark:hover:bg-slate-600 focus:ring-blue-500 focus:outline-none transition-colors"
      >
        Clear All Filters
      </button>
    </div>
  );
};

export default SearchFilters;