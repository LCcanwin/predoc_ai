import React from 'react'
import { TenInquiryOptions } from './TenInquiryOptions'
import { ConfirmationQuestions } from './ConfirmationQuestions'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  isLoading?: boolean
  isThinking?: boolean
  phase?: string
}

interface MessageBubbleProps {
  message: Message
  onOptionsSelect?: (selections: { dimensionId: number; value: string }[]) => void
  onConfirmQuestions?: (answers: string[]) => void
}

// Parse content to detect if it's showing ten inquiry options
const isTenInquiryOptions = (content: string): boolean => {
  return content.includes('【中医十问歌】') ||
    content.includes('请选择您有的症状') ||
    (content.includes('1. 寒热') && content.includes('请选择'));
}

// Parse content to detect if it's asking for confirmation questions
const isConfirmationQuestions = (content: string): boolean => {
  return content.includes('请确认以下') ||
    (content.includes('1.') && content.includes('?'));
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  onOptionsSelect,
  onConfirmQuestions,
}) => {
  const isUser = message.role === 'user'
  const timeStr = message.timestamp.toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit'
  })

  // Don't render thinking state visually - just show loading
  if (message.isThinking) {
    return (
      <div className={`message-wrapper ${message.role}`}>
        <div className={`message-avatar ${message.role}`}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z"/>
            <circle cx="12" cy="12" r="3"/>
          </svg>
        </div>
        <div className="message-content">
          <div className="message-text thinking">
            <span className="thinking-dots">
              <span></span><span></span><span></span>
            </span>
          </div>
          <div className="message-time">{timeStr}</div>
        </div>
      </div>
    )
  }

  // If it's an assistant message with options, render the options component
  if (!isUser && isTenInquiryOptions(message.content) && onOptionsSelect) {
    const [introRaw] = message.content.split('【中医十问歌】')
    const intro = introRaw
      .replace('【已理解您的意图】', '')
      .trim()

    return (
      <div className={`message-wrapper ${message.role}`}>
        <div className={`message-avatar ${message.role}`}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z"/>
            <circle cx="12" cy="12" r="3"/>
          </svg>
        </div>
        <div className="message-content">
          {intro && <div className="message-intro">{intro}</div>}
          <div className="message-text">
            <TenInquiryOptions
              onSelect={onOptionsSelect}
              disabled={message.isLoading}
            />
          </div>
          <div className="message-time">{timeStr}</div>
        </div>
      </div>
    )
  }

  // If it's an assistant message with confirmation questions
  if (!isUser && isConfirmationQuestions(message.content) && onConfirmQuestions) {
    // Extract the intro text before questions
    const lines = message.content.split('\n')
    const introLines: string[] = []
    const questionLines: string[] = []

    let foundQuestions = false
    for (const line of lines) {
      if (line.match(/^\d+[.．、]/)) {
        foundQuestions = true
      }
      if (foundQuestions) {
        questionLines.push(line)
      } else if (line.trim()) {
        introLines.push(line)
      }
    }

    const intro = introLines.join('\n').replace('【已理解您的意图】', '').replace('【已理解您的意图】\n', '')
    const questions = questionLines
      .filter((l) => l.match(/^\d+[.．、]/))
      .map((l) => l.replace(/^\d+[.．、]\s*/, '').replace(/\s*\[.*?\]\s*$/, ''))

    return (
      <div className={`message-wrapper ${message.role}`}>
        <div className={`message-avatar ${message.role}`}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z"/>
            <circle cx="12" cy="12" r="3"/>
          </svg>
        </div>
        <div className="message-content">
          {intro && <div className="message-intro">{intro}</div>}
          {questions.length > 0 && (
            <ConfirmationQuestions
              questions={questions}
              onConfirm={onConfirmQuestions}
              disabled={message.isLoading}
            />
          )}
          <div className="message-time">{timeStr}</div>
        </div>
      </div>
    )
  }

  // Regular message bubble
  return (
    <div className={`message-wrapper ${message.role}`}>
      <div className={`message-avatar ${message.role}`}>
        {isUser ? (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
            <circle cx="12" cy="7" r="4"/>
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z"/>
            <circle cx="12" cy="12" r="3"/>
          </svg>
        )}
      </div>
      <div className="message-content">
        <div className={`message-text ${message.isLoading ? 'loading-dots' : ''}`}>
          {message.content}
          {message.isLoading && <span className="loading-cursor">▊</span>}
        </div>
        <div className="message-time">{timeStr}</div>
      </div>
    </div>
  )
}
