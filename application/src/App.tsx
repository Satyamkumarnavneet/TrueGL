import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import ResultsPage from './pages/ResultsPage';
import { SearchProvider } from './context/SearchContext';

function App() {
  return (
    <Router>
      <SearchProvider>
        <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-100 transition-colors duration-200">
          <Header />
          <main className="flex-grow">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/search" element={<ResultsPage />} />
            </Routes>
          </main>
          <Footer />
        </div>
      </SearchProvider>
    </Router>
  );
}

export default App;