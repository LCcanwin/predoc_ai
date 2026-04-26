import React from 'react'

interface ThinkingIndicatorProps {
  status: 'thinking' | 'reflection' | 'generating'
  message?: string
}

const statusText = {
  thinking: '正在理解您的意图',
  reflection: '正在分析症状',
  generating: '正在生成病例',
}

export const ThinkingIndicator: React.FC<ThinkingIndicatorProps> = ({ status, message }) => {
  return (
    <div className="thinking-wrapper">
      <div className="thinking-avatar">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z"/>
          <circle cx="12" cy="12" r="3"/>
        </svg>
      </div>
      <div className="thinking-text">
        {message || statusText[status]}
      </div>
      <div className="thinking-dots">
        <span></span>
        <span></span>
        <span></span>
      </div>
    </div>
  )
}
