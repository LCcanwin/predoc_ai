import React, { useState, useRef, useEffect } from 'react'
import { MessageBubble } from './MessageBubble'
import { CaseOutput } from './CaseOutput'
import { useChatStream } from '../hooks/useChatStream'
import { api, type User } from '../services/api'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  isLoading?: boolean
  isThinking?: boolean
  phase?: string
}

interface ChatWindowProps {
  token: string | null
  user: User | null
  onRequireAuth: () => void
  onLogout: () => void
}

const welcomeMessage: Message = {
  id: 'welcome',
  role: 'assistant',
  content: '您好，这里是智能预问诊助手。\n\n请直接描述您当前最主要的不适，例如持续时间、出现时段、诱因或伴随症状。我会先提取关键信息，再请您补充确认，最后生成诊断结论。',
  timestamp: new Date(),
}

export const ChatWindow: React.FC<ChatWindowProps> = ({ token, user, onRequireAuth, onLogout }) => {
  const [messages, setMessages] = useState<Message[]>([welcomeMessage])
  const [input, setInput] = useState('')
  const [threadId, setThreadId] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [showCase, setShowCase] = useState(false)
  const [generatedCase, setGeneratedCase] = useState<string | null>(null)
  const [isGeneratingCase, setIsGeneratingCase] = useState(false)
  const [pendingOptions, setPendingOptions] = useState<number[] | null>(null)
  const [pendingConfirm, setPendingConfirm] = useState<string[] | null>(null)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { sendMessage, startConsultation } = useChatStream(token)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    if (!token) {
      setThreadId(null)
      setMessages([welcomeMessage])
      return
    }

    const init = async () => {
      try {
        const newThreadId = await startConsultation()
        setThreadId(newThreadId)
      } catch (error) {
        console.error('Failed to initialize:', error)
      }
    }
    init()
  }, [startConsultation])

  // Handle option selection from TenInquiryOptions
  const handleOptionsSelect = (selections: { dimensionId: number; value: string }[]) => {
    // Format selections as readable string
    const dimensionNames: Record<number, string> = {
      1: '寒热', 2: '汗', 3: '头身', 4: '便溏', 5: '饮食',
      6: '胸腹', 7: '耳目', 8: '口渴', 9: '睡眠', 10: '舌脉',
    }
    const optionString = selections
      .map(s => `${dimensionNames[s.dimensionId] || s.dimensionId}: ${s.value}`)
      .join('; ')

    setInput(optionString)
    setTimeout(() => {
      handleSendWithInput(optionString)
    }, 100)
  }

  // Handle confirmation questions answer
  const handleConfirmAnswers = (answers: string[]) => {
    const answerString = answers.join(',')
    setInput(answerString)
    setTimeout(() => {
      handleSendWithInput(answerString)
    }, 100)
  }

  // Send message with custom input (for option selections)
  const handleSendWithInput = async (inputContent: string) => {
    if (!inputContent.trim() || isLoading) return
    if (!token || !threadId) {
      onRequireAuth()
      return
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputContent.trim(),
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setIsLoading(true)
    setShowCase(false)
    setGeneratedCase(null)
    setIsGeneratingCase(false)
    setPendingOptions(null)
    setPendingConfirm(null)

    const assistantMessageId = (Date.now() + 1).toString()
    const loadingMessage: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isLoading: true,
      isThinking: true,
    }

    setMessages((prev) => [...prev, loadingMessage])

    try {
      await sendMessage(threadId, inputContent.trim(), {
        onThinking: (_status, _message) => {},
        onMessage: (content, _isComplete) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMessageId
                ? { ...m, content, isLoading: false, isThinking: false }
                : m
            )
          )
        },
        onComplete: async (isComplete: boolean) => {
          if (!isComplete) {
            setIsLoading(false)
            return
          }
          try {
            setIsGeneratingCase(true)
            const data = await api.getCase(threadId, token)
            setGeneratedCase(data.case)
            setShowCase(true)
          } catch (e) {
            console.error('Failed to fetch case:', e)
          } finally {
            setIsGeneratingCase(false)
          }
          setIsLoading(false)
        },
        onError: (error) => {
          console.error('Error:', error)
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMessageId
                ? { ...m, content: '抱歉，出现了一些问题，请重试。', isLoading: false }
                : m
            )
          )
          setIsLoading(false)
          setIsGeneratingCase(false)
        },
      })
    } catch (error) {
      console.error('Failed to send message:', error)
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMessageId
            ? { ...m, content: '抱歉，出现了一些问题，请重试。', isLoading: false }
            : m
        )
      )
      setIsLoading(false)
      setIsGeneratingCase(false)
    }
  }

  // Regular send handler
  const handleSend = () => {
    if (!input.trim()) return
    handleSendWithInput(input.trim())
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-avatar">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M4 5h16"/>
            <path d="M4 12h16"/>
            <path d="M4 19h10"/>
          </svg>
        </div>
        <div className="header-info">
          <div className="header-title">智能预问诊</div>
          <div className="header-subtitle">症状采集 · 辨证分析 · 诊断结论</div>
        </div>
        <div className="header-status">
          <span className="status-dot"></span>
          {user ? user.username : '未登录'}
        </div>
        {user ? (
          <button className="logout-button" onClick={onLogout}>退出</button>
        ) : (
          <button className="logout-button" onClick={onRequireAuth}>登录</button>
        )}
      </header>

      <div className="messages-container">
        {messages.map((msg) => (
          <MessageBubble
            key={msg.id}
            message={msg}
            onOptionsSelect={handleOptionsSelect}
            onConfirmQuestions={handleConfirmAnswers}
          />
        ))}

        {isGeneratingCase && (
          <div className="case-container case-loading-container">
            <div className="case-header">
              <div className="case-icon case-icon-loading">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                  <polyline points="14 2 14 8 20 8"/>
                  <line x1="16" y1="13" x2="8" y2="13"/>
                  <line x1="16" y1="17" x2="8" y2="17"/>
                </svg>
              </div>
              <div>
                <div className="case-title loading-dots">报告正在生成中</div>
                <div className="case-subtitle">正在整理症状摘要与分析结论</div>
              </div>
            </div>
            <div className="case-loading-body">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}

        {showCase && generatedCase && !isGeneratingCase && (
          <div className="case-container">
            <div className="case-header">
              <div className="case-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                  <polyline points="14 2 14 8 20 8"/>
                  <line x1="16" y1="13" x2="8" y2="13"/>
                  <line x1="16" y1="17" x2="8" y2="17"/>
                  <polyline points="10 9 9 9 8 9"/>
                </svg>
              </div>
              <div>
                <div className="case-title">诊断结论</div>
                <div className="case-subtitle">症状摘要 · 辨证分析 · 调理建议</div>
              </div>
            </div>
            <CaseOutput content={generatedCase} />
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="input-container">
        <div className="input-wrapper">
          <textarea
            className="input-field"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              pendingOptions
                ? '已选择选项，正在发送...'
                : pendingConfirm
                  ? '请输入您的答案...'
                  : '请描述您的症状...'
            }
            disabled={isLoading}
            rows={1}
          />
          <button
            className="send-button"
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="22" y1="2" x2="11" y2="13"/>
              <polygon points="22 2 15 22 11 13 2 9 22 2"/>
            </svg>
          </button>
        </div>
      </div>
    </div>
  )
}
