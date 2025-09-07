import React from 'react';
import { TruthScoreCategory } from '../types';

interface TruthScoreBadgeProps {
  score: number;
  bias?: string; // Add bias prop
  size?: 'sm' | 'md' | 'lg';
}

const TruthScoreBadge: React.FC<TruthScoreBadgeProps> = ({ score, bias, size = 'md' }) => {
  // Determine color and category based on score
  let category: TruthScoreCategory;
  let bgColor: string;
  let textColor: string;
  
  if (score >= 70) {
    category = 'high';
    bgColor = 'bg-green-100 dark:bg-green-900';
    textColor = 'text-green-800 dark:text-green-100';
  } else if (score >= 50) {
    category = 'medium';
    bgColor = 'bg-yellow-100 dark:bg-yellow-900';
    textColor = 'text-yellow-800 dark:text-yellow-100';
  } else {
    category = 'low';
    bgColor = 'bg-red-100 dark:bg-red-900';
    textColor = 'text-red-800 dark:text-red-100';
  }
  
  // Size classes
  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-sm px-2.5 py-1',
    lg: 'text-base px-3 py-1.5'
  };
  
  return (
    <div 
      className={`inline-flex items-center rounded-full ${bgColor} ${textColor} ${sizeClasses[size]} font-medium`}
      title={`Truth Score: ${score}% - ${category.charAt(0).toUpperCase() + category.slice(1)} reliability`}
    >
      <span className="mr-1 font-bold">{score}%</span>
      {bias && <span className="ml-1 text-xs">({bias})</span>}
    </div>
  );
};

export default TruthScoreBadge;