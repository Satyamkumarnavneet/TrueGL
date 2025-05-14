// This is a suggestion based on typical implementation. Adjust according to your actual file structure.

// ...existing imports...

export const search = async (query: string, offset: number = 0, limit: number = 10) => {
  try {
    const response = await fetch(`https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&offset=${offset}&limit=${limit}&search_lang=en&safesearch=moderate`, {
      headers: {
        'Accept': 'application/json',
        'X-Subscription-Token': process.env.REACT_APP_BRAVE_API_KEY || ''
      }
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error during search: ${response.statusText} for url '${response.url}' For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/${response.status}`);
    }
    
    const data = await response.json();
    return data.web?.results || [];
  } catch (error) {
    console.error('Search API error:', error);
    throw error;
  }
};

// ...rest of the code...
