const API_BASE = '/api/consultation'
const AUTH_BASE = '/api/auth'
const TOKEN_KEY = 'predoc_auth_token'
const USER_KEY = 'predoc_auth_user'

export interface ConsultationResponse {
  thread_id: string
  created_at: string
  memory_context?: string
}

export interface CaseResponse {
  case: string
}

export interface User {
  id: string
  username: string
  created_at?: string
}

export interface AuthResponse {
  access_token: string
  token_type: string
  user: User
}

export const authStore = {
  getToken(): string | null {
    return localStorage.getItem(TOKEN_KEY)
  },
  getUser(): User | null {
    const raw = localStorage.getItem(USER_KEY)
    if (!raw) return null
    try {
      return JSON.parse(raw)
    } catch {
      return null
    }
  },
  setAuth(auth: AuthResponse) {
    localStorage.setItem(TOKEN_KEY, auth.access_token)
    localStorage.setItem(USER_KEY, JSON.stringify(auth.user))
  },
  clear() {
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
  },
}

const authHeaders = (token?: string | null) => ({
  'Content-Type': 'application/json',
  ...(token ? { Authorization: `Bearer ${token}` } : {}),
})

export const api = {
  async register(username: string, password: string): Promise<AuthResponse> {
    const response = await fetch(`${AUTH_BASE}/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    })

    if (!response.ok) {
      const data = await response.json().catch(() => ({}))
      throw new Error(data.detail || '注册失败')
    }

    return response.json()
  },

  async login(username: string, password: string): Promise<AuthResponse> {
    const response = await fetch(`${AUTH_BASE}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    })

    if (!response.ok) {
      const data = await response.json().catch(() => ({}))
      throw new Error(data.detail || '登录失败')
    }

    return response.json()
  },

  async me(token: string): Promise<User> {
    const response = await fetch(`${AUTH_BASE}/me`, {
      headers: authHeaders(token),
    })

    if (!response.ok) {
      throw new Error('登录已失效')
    }

    return response.json()
  },

  async startConsultation(userName?: string, token?: string | null): Promise<ConsultationResponse> {
    const response = await fetch(`${API_BASE}/start`, {
      method: 'POST',
      headers: authHeaders(token),
      body: JSON.stringify({ user_name: userName }),
    })

    if (!response.ok) {
      throw new Error('Failed to start consultation')
    }

    return response.json()
  },

  async getCase(threadId: string, token?: string | null): Promise<CaseResponse> {
    const response = await fetch(`${API_BASE}/${threadId}/case`, {
      headers: authHeaders(token),
    })

    if (!response.ok) {
      throw new Error('Failed to get case')
    }

    return response.json()
  },
}
