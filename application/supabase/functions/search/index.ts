import { createClient } from 'npm:@supabase/supabase-js@2.39.7';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { query } = await req.json();
    
    if (!query) {
      throw new Error('Query parameter is required');
    }

    const GOOGLE_API_KEY = Deno.env.get('GOOGLE_API_KEY');
    const GOOGLE_CX = Deno.env.get('GOOGLE_CX');
    
    if (!GOOGLE_API_KEY || !GOOGLE_CX) {
      throw new Error('Search configuration is missing');
    }

    const searchUrl = `https://www.googleapis.com/customsearch/v1?key=${GOOGLE_API_KEY}&cx=${GOOGLE_CX}&q=${encodeURIComponent(query)}`;
    
    const response = await fetch(searchUrl);
    const data = await response.json();

    // Transform Google results into our format
    const results = data.items?.map((item: any) => ({
      id: item.cacheId || item.link,
      title: item.title,
      url: item.link,
      snippet: item.snippet,
      truthScore: await calculateTruthScore(item.link, item.snippet),
      dateIndexed: new Date().toISOString()
    })) || [];

    return new Response(JSON.stringify(results), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 400,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }
});

async function calculateTruthScore(url: string, content: string): Promise<number> {
  // Initialize Supabase client
  const supabaseClient = createClient(
    Deno.env.get('SUPABASE_URL') ?? '',
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
  );

  try {
    // Use embeddings to analyze content
    const embeddingResponse = await supabaseClient.functions.invoke('generate-embeddings', {
      body: { text: content }
    });

    // For now, return a random score between 30 and 95
    // This will be replaced with actual ML-based scoring
    return Math.floor(Math.random() * (95 - 30 + 1)) + 30;
  } catch (error) {
    console.error('Error calculating truth score:', error);
    return 50; // Default score on error
  }
}