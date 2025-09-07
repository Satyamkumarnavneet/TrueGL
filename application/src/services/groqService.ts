import axios from 'axios';

interface GroqResponse {
  choices: {
    message: {
      content: string;
    };
  }[];
}

/**
 * Generate a justification for the truthfulness of content based on the URL, title, and snippet.
 * First attempts to use the Groq API, and falls back to local analysis if the API call fails.
 */
export const getJustification = async (url: string, title: string, snippet: string): Promise<string> => {
  try {
    // Attempt to get justification from Groq API
    const apiJustification = await getJustificationFromApi(url, title, snippet);
    return apiJustification;
  } catch (error) {
    console.error('Error getting justification from Groq API:', error);
    // Fallback to local analysis if API call fails
    return generateLocalJustification(url, title, snippet);
  }
};

/**
 * Get justification from Groq API
 */
async function getJustificationFromApi(url: string, title: string, snippet: string): Promise<string> {
  const prompt = `
    You are evaluating the truthfulness of a web page. I'll provide you with the URL, title, and a snippet from the content.
    Please analyze this information and provide a detailed justification about why this content might be reliable or unreliable.
    Consider factors like source reputation, factual accuracy, bias indicators, and presentation style.
    
    URL: ${url}
    Title: ${title}
    Content snippet: ${snippet}
    
    Provide your truth justification in a concise paragraph:
  `;

  const response = await axios.post<GroqResponse>(
    'https://api.groq.com/openai/v1/chat/completions',
    {
      model: 'llama3-70b-8192',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.5
    },
    {
      headers: {
        'Authorization': `Bearer ${import.meta.env.VITE_GROQ_API_KEY}`,
        'Content-Type': 'application/json'
      }
    }
  );

  return response.data.choices[0].message.content;
}

