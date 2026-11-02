const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000')
  .replace(/\/+$/, '')
  .replace(/\/api$/i, '');

export async function request<T = any>(path: string, options: RequestInit = {}, signal?: AbortSignal): Promise<T> {
  const headers = new Headers(options.headers || {});
  const token = localStorage.getItem('rl_token');
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  const response = await fetch(`${API_BASE_URL}/api${normalizedPath}`, { ...options, headers, signal });
  if (response.status === 401 && token) { localStorage.removeItem('rl_token'); window.location.reload(); }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload as T;
}

export const api = {
  health: (signal?: AbortSignal) => request('/health', {}, signal),
  documents: (signal?: AbortSignal) => request('/documents', {}, signal),
  vendors: (signal?: AbortSignal) => request('/vendors', {}, signal),
  evaluations: (signal?: AbortSignal) => request('/evaluations', {}, signal),
  reviews: (signal?: AbortSignal) => request('/reviews', {}, signal),
  audit: (signal?: AbortSignal) => request('/audit', {}, signal),
  models: (signal?: AbortSignal) => request('/models/status', {}, signal),
  clauses: (id: number, signal?: AbortSignal) => request(`/documents/${id}/clauses`, {}, signal),
  upload: (file: File) => { const body = new FormData(); body.append('file', file); return request('/documents/upload', { method: 'POST', body }); },
  removeDocument: (id: number) => request(`/documents/${id}`, { method: 'DELETE' }),
  ask: (question: string, document_ids?: number[]) => request('/assistant/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, document_ids }) }),
  assess: (form: unknown) => request('/vendors/assess', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(form) }),
  review: (id: number, data: {status: string; reviewer: string; note: string}) => request(`/reviews/${id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }),
  login: (username: string, password: string) => request('/auth/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username, password }) }),
};
