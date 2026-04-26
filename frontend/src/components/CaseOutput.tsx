import React from 'react'

interface CaseOutputProps {
  content: string
}

export const CaseOutput: React.FC<CaseOutputProps> = ({ content }) => {
  const renderContent = (text: string) => {
    const lines = text.split('\n')
    const elements: React.ReactNode[] = []

    lines.forEach((line, index) => {
      if (line.startsWith('# ')) {
        elements.push(
          <h1 key={index}>{line.substring(2)}</h1>
        )
      } else if (line.startsWith('## ')) {
        elements.push(
          <h2 key={index}>{line.substring(3)}</h2>
        )
      } else if (line.startsWith('- ')) {
        elements.push(
          <li key={index}>{line.substring(2)}</li>
        )
      } else if (line.trim() === '') {
        elements.push(<br key={index} />)
      } else {
        elements.push(
          <p key={index}>{line}</p>
        )
      }
    })

    return elements
  }

  return (
    <div className="case-content">
      {renderContent(content)}
    </div>
  )
}
