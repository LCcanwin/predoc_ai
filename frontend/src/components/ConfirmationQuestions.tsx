import React, { useState } from 'react'

interface ConfirmationQuestionsProps {
  questions: string[]
  onConfirm: (answers: string[]) => void
  disabled?: boolean
}

export const ConfirmationQuestions: React.FC<ConfirmationQuestionsProps> = ({
  questions,
  onConfirm,
  disabled,
}) => {
  const [answers, setAnswers] = useState<string[]>(questions.map(() => ''))

  const handleAnswerChange = (index: number, value: string) => {
    const newAnswers = [...answers]
    newAnswers[index] = value
    setAnswers(newAnswers)
  }

  const handleConfirm = () => {
    const allAnswered = answers.every((a) => a.trim() !== '')
    if (allAnswered) {
      onConfirm(answers)
    }
  }

  const allAnswered = answers.every((a) => a.trim() !== '')

  return (
    <div className="confirmation-container">
      <div className="confirmation-title">请确认以下问题</div>
      <div className="confirmation-list">
        {questions.map((question, index) => (
          <div key={index} className="confirmation-item">
            <div className="confirmation-question">
              <span className="question-number">{index + 1}</span>
              <span className="question-text">{question}</span>
            </div>
            <input
              type="text"
              className="confirmation-input"
              placeholder="请输入您的答案..."
              value={answers[index]}
              onChange={(e) => handleAnswerChange(index, e.target.value)}
              disabled={disabled}
            />
          </div>
        ))}
      </div>
      <button
        className="confirmation-confirm"
        onClick={handleConfirm}
        disabled={!allAnswered || disabled}
      >
        确认答案
      </button>
    </div>
  )
}

// Parse confirmation questions from content
export const parseConfirmationQuestions = (content: string): string[] | null => {
  // Look for numbered questions in the content
  const lines = content.split('\n')
  const questions: string[] = []

  for (const line of lines) {
    // Match patterns like "1. 问题内容" or "1. 问题内容？[选项]"
    const match = line.match(/^\d+[.．、]\s*(.+)/)
    if (match) {
      let question = match[1].trim()
      // Remove option brackets like [选项] at the end
      question = question.replace(/\s*\[.*?\]\s*$/, '')
      if (question) questions.push(question)
    }
  }

  return questions.length > 0 ? questions : null
}
