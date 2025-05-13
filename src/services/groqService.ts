import axios from 'axios';

interface GroqResponse {
  choices: {
    message: {
      content: string;
    };
  }[];
}

export const getJustification = async (url: string, title: string, snippet: string): Promise<string> => {
  try {
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
        model: 'llama-3.1-8b-instant',
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.5
      },
      {
        headers: {
          'Authorization': `Bearer ${process.env.REACT_APP_GROQ_API_KEY || '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'}`,
          'Content-Type': 'application/json'
        }
      }
    );

    return response.data.choices[0].message.content;
  } catch (error) {
    console.error('Error getting justification from Groq:', error);
    return 'Unable to generate justification at this time. Please try again later.';
  }
};
