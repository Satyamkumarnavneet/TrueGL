export interface SearchResult {
  id: string;
  title: string;
  url: string;
  snippet: string;
  truthScore: number;
  dateIndexed: string;
}

export type TruthScoreCategory = 'low' | 'medium' | 'high';