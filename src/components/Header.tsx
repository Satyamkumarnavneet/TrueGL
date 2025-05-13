import React, { useState, useEffect } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Search, Moon, Sun, Menu, X } from 'lucide-react';

const Header = () => {
  // Simplify dark mode implementation
  const [darkMode, setDarkMode] = useState(() => {
    return document.documentElement.classList.contains('dark');
  });
  
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const [inputValue, setInputValue] = useState('');
  const isHomePage = location.pathname === '/';
  
  // Initialize dark mode on component mount
  useEffect(() => {
    // Check localStorage first
    const savedTheme = localStorage.getItem('theme');
    
    if (savedTheme === 'dark' || 
       (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      document.documentElement.classList.add('dark');
      setDarkMode(true);
    } else {
      document.documentElement.classList.remove('dark');
      setDarkMode(false);
    }
  }, []);

  // Toggle dark mode function - simplified
  const toggleDarkMode = () => {
    if (darkMode) {
      // Switch to light mode
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
      setDarkMode(false);
      console.log('Switched to light mode');
    } else {
      // Switch to dark mode
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
      setDarkMode(true);
      console.log('Switched to dark mode');
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim()) {
      navigate(`/search?q=${encodeURIComponent(inputValue.trim())}`);
    }
  };

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <header className={`sticky top-0 z-10 transition-colors duration-200 ${isHomePage ? 'bg-transparent' : 'bg-white dark:bg-slate-800 shadow-sm'}`}>
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          <Link to="/" className="flex items-center space-x-2">
            <Search className="h-6 w-6 text-blue-600 dark:text-blue-400" />
            <span className="font-bold text-xl text-blue-600 dark:text-blue-400">TrueGL</span>
          </Link>

          {!isHomePage && (
            <form 
              onSubmit={handleSubmit}
              className="hidden md:flex flex-grow mx-8 max-w-2xl"
            >
              <div className="relative w-full">
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="Search for the truth..."
                  className="w-full p-2 pl-10 rounded-full border border-gray-300 dark:border-gray-600 
                           dark:bg-slate-700 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent
                           transition-all duration-200"
                />
                <Search className="absolute left-3 top-2.5 h-4 w-4 text-gray-500 dark:text-gray-400" />
              </div>
              <button
                type="submit"
                className="ml-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-full
                          transition-colors duration-200 flex items-center justify-center"
              >
                Search
              </button>
            </form>
          )}

          <button
            onClick={toggleDarkMode}
            className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-slate-700 
                      transition-colors duration-200 mr-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            aria-label="Toggle dark mode"
            title={darkMode ? "Switch to light mode" : "Switch to dark mode"}
          >
            {darkMode ? 
              <Sun className="h-5 w-5 text-yellow-400" /> : 
              <Moon className="h-5 w-5 text-gray-600" />
            }
          </button>

          <div className="md:hidden">
            <button
              onClick={toggleMenu}
              className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-slate-700 
                        transition-colors duration-200"
              aria-label="Menu"
            >
              {isMenuOpen ? 
                <X className="h-5 w-5" /> : 
                <Menu className="h-5 w-5" />
              }
            </button>
          </div>
        </div>

        {/* Mobile menu */}
        {isMenuOpen && !isHomePage && (
          <div className="md:hidden mt-3 pb-2">
            <form onSubmit={handleSubmit} className="flex flex-col space-y-2">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Search for the truth..."
                className="w-full p-2 rounded-lg border border-gray-300 dark:border-gray-600 
                         dark:bg-slate-700 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <button
                type="submit"
                className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg
                          transition-colors duration-200"
              >
                Search
              </button>
            </form>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;