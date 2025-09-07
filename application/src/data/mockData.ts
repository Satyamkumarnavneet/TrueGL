import { SearchResult } from '../types';

export const mockSearchResults: SearchResult[] = [
  {
    id: '1',
    title: 'Understanding Climate Change: Facts and Myths',
    url: 'https://example.com/climate-facts',
    snippet: 'Climate change is a global phenomenon characterized by shifts in temperature, precipitation patterns, and increased frequency of extreme weather events...',
    truthScore: 92,
    dateIndexed: '2025-03-15T10:30:00Z'
  },
  {
    id: '2',
    title: 'Debunking Common Health Misconceptions',
    url: 'https://healthsite.org/misconceptions',
    snippet: 'Many popular health claims lack scientific evidence. This article examines common health myths and presents the current scientific consensus...',
    truthScore: 87,
    dateIndexed: '2025-03-12T14:25:00Z'
  },
  {
    id: '3',
    title: 'The Economic Impact of Artificial Intelligence',
    url: 'https://techreview.com/ai-economy',
    snippet: 'Artificial intelligence is transforming industries globally. We analyze the potential economic benefits and challenges as AI adoption increases...',
    truthScore: 78,
    dateIndexed: '2025-03-10T09:15:00Z'
  },
  {
    id: '4',
    title: 'Alternative Medicine: Exploring the Evidence',
    url: 'https://alternativemedicine.com/evidence',
    snippet: 'This comprehensive guide evaluates various alternative medicine practices and their effectiveness according to scientific research...',
    truthScore: 45,
    dateIndexed: '2025-03-08T11:20:00Z'
  },
  {
    id: '5',
    title: 'Political Analysis: Current Administration Policies',
    url: 'https://politicalsite.com/analysis',
    snippet: 'Our team of experts examines the policies implemented by the current administration and their potential long-term effects...',
    truthScore: 62,
    dateIndexed: '2025-03-07T16:45:00Z'
  },
  {
    id: '6',
    title: 'Breaking News: Global Economic Summit Results',
    url: 'https://worldnews.com/economic-summit',
    snippet: 'World leaders reached consensus on new trade agreements at yesterday\'s summit. We analyze the implications for global markets...',
    truthScore: 81,
    dateIndexed: '2025-03-05T21:30:00Z'
  },
  {
    id: '7',
    title: 'Conspiracy Theories Debunked: Separating Fact from Fiction',
    url: 'https://factchecker.org/conspiracies',
    snippet: 'We investigate popular conspiracy theories and evaluate them against available evidence and expert analysis...',
    truthScore: 94,
    dateIndexed: '2025-02-28T13:10:00Z'
  },
  {
    id: '8',
    title: 'New Study Claims Revolutionary Weight Loss Method',
    url: 'https://diettrends.com/new-method',
    snippet: 'A recently published study suggests a breakthrough approach to weight loss that challenges conventional nutritional wisdom...',
    truthScore: 31,
    dateIndexed: '2025-02-25T10:05:00Z'
  }
];