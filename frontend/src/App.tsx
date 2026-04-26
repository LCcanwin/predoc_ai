import { useEffect, useState } from 'react'
import { ChatWindow } from './components/ChatWindow'
import { api, authStore, type User } from './services/api'

function App() {
  const [token, setToken] = useState<string | null>(() => authStore.getToken())
  const [user, setUser] = useState<User | null>(() => authStore.getUser())
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [showAuth, setShowAuth] = useState(false)

  useEffect(() => {
    if (!token) return
    api.me(token)
      .then((profile) => {
        setUser(profile)
      })
      .catch(() => {
        authStore.clear()
        setToken(null)
        setUser(null)
      })
  }, [token])

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    setError('')
    setIsSubmitting(true)
    try {
      const auth = mode === 'login'
        ? await api.login(username, password)
        : await api.register(username, password)
      authStore.setAuth(auth)
      setToken(auth.access_token)
      setUser(auth.user)
      setUsername('')
      setPassword('')
      setShowAuth(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : '认证失败')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleLogout = () => {
    authStore.clear()
    setToken(null)
    setUser(null)
  }

  return (
    <>
      <ChatWindow
        token={token}
        user={user}
        onRequireAuth={() => {
          setError('')
          setShowAuth(true)
        }}
        onLogout={handleLogout}
      />

      {showAuth && !user && (
        <div className="auth-overlay" role="dialog" aria-modal="true">
          <form className="auth-panel auth-modal" onSubmit={handleSubmit}>
            <button
              type="button"
              className="auth-close"
              aria-label="关闭"
              onClick={() => setShowAuth(false)}
            >
              ×
            </button>

            <div className="auth-brand">
              <div className="auth-mark">诊</div>
              <div>
                <h1>{mode === 'login' ? '登录后继续问诊' : '注册账号'}</h1>
                <p>问诊记录和近期症状记忆会保存到您的账号。</p>
              </div>
            </div>

            <div className="auth-tabs">
              <button
                type="button"
                className={mode === 'login' ? 'active' : ''}
                onClick={() => setMode('login')}
              >
                登录
              </button>
              <button
                type="button"
                className={mode === 'register' ? 'active' : ''}
                onClick={() => setMode('register')}
              >
                注册
              </button>
            </div>

            <label className="auth-field">
              <span>用户名</span>
              <input
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                minLength={3}
                maxLength={50}
                autoComplete="username"
                required
                autoFocus
              />
            </label>

            <label className="auth-field">
              <span>密码</span>
              <input
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                type="password"
                minLength={6}
                maxLength={128}
                autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                required
              />
            </label>

            {error && <div className="auth-error">{error}</div>}

            <button className="auth-submit" type="submit" disabled={isSubmitting}>
              {isSubmitting ? '处理中...' : mode === 'login' ? '登录并继续' : '注册并继续'}
            </button>
          </form>
        </div>
      )}
    </>
  )
}

export default App
