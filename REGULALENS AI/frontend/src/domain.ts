export type CitationSource = { id: string; filename?: string; page?: number };
export function auditCitations(answer: string, sources: CitationSource[]) {
  const cited = [...answer.matchAll(/\[(S\d+)\]/g)].map((match) => match[1]);
  const allowed = new Set(sources.map((source) => source.id));
  return {
    cited,
    invalid: [...new Set(cited.filter((id) => !allowed.has(id)))],
    uncited: sources.filter((source) => !cited.includes(source.id)).map((source) => source.id),
  };
}

export function riskDisplay(score: number) {
  const safe = Number.isFinite(score) ? Math.max(0, Math.min(100, score)) : 0;
  return { score: safe, percent: `${safe.toFixed(1)}%`, level: safe >= 67 ? 'high' : safe >= 34 ? 'medium' : 'low' };
}
