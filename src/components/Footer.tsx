import React from 'react';
import { Search, Github, Info, Shield } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="bg-white dark:bg-slate-800 border-t border-gray-200 dark:border-slate-700 py-6 mt-8">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="flex items-center mb-4 md:mb-0">
            <Search className="h-5 w-5 text-blue-600 dark:text-blue-400 mr-2" />
            <span className="font-bold text-lg text-blue-600 dark:text-blue-400">TrueGL</span>
          </div>
          
          <div className="flex flex-wrap justify-center gap-x-6 gap-y-2 text-sm text-gray-600 dark:text-gray-300">
            <a href="#" className="hover:text-blue-600 dark:hover:text-blue-400 flex items-center">
              <Info className="h-4 w-4 mr-1" />
              <span>About</span>
            </a>
            <a href="#" className="hover:text-blue-600 dark:hover:text-blue-400 flex items-center">
              <Shield className="h-4 w-4 mr-1" />
              <span>Privacy</span>
            </a>
            <a href="#" className="hover:text-blue-600 dark:hover:text-blue-400 flex items-center">
              <Github className="h-4 w-4 mr-1" />
              <span>GitHub</span>
            </a>
          </div>
        </div>
        
        <div className="mt-6 text-center text-xs text-gray-500 dark:text-gray-400">
          <p>© {new Date().getFullYear()} TrueGL. All rights reserved.</p>
          <p className="mt-1">
            Helping you find reliable information since 2025.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;