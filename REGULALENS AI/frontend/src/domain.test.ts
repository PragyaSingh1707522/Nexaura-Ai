import { describe, expect, it } from 'vitest';
import { auditCitations, riskDisplay } from './domain';

describe('citation display checks', () => {
  it('flags source IDs that were not returned by retrieval', () => {
    expect(auditCitations('Retention is 90 days [S8].', [{ id: 'S1' }]).invalid).toEqual(['S8']);
  });
  it('handles repeated valid citations and uncited sources', () => {
    expect(auditCitations('See [S1], then [S1].', [{ id: 'S1' }, { id: 'S2' }])).toEqual({ cited: ['S1', 'S1'], invalid: [], uncited: ['S2'] });
  });
});

describe('risk display', () => {
  it('clamps the score and returns its risk band', () => {
    expect(riskDisplay(120)).toEqual({ score: 100, percent: '100.0%', level: 'high' });
    expect(riskDisplay(12).level).toBe('low');
  });
});
