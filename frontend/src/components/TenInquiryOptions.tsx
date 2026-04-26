import React, { useState } from 'react'

interface TenInquiryOptionsProps {
  onSelect: (selections: { dimensionId: number; value: string }[]) => void
  disabled?: boolean
}

// Ten inquiry dimensions with detailed options
const TEN_INQUIRY = [
  {
    id: 1,
    name: '寒热',
    icon: '01',
    options: ['正常', '畏寒', '发热', '寒热往来', '手脚冰凉', '其他'],
  },
  {
    id: 2,
    name: '汗',
    icon: '02',
    options: ['正常', '有汗', '无汗', '盗汗', '自汗', '冷汗', '其他'],
  },
  {
    id: 3,
    name: '头身',
    icon: '03',
    options: ['正常', '头痛', '头晕', '身痛', '腰痛', '肩背痛', '乏力', '其他'],
  },
  {
    id: 4,
    name: '便溏',
    icon: '04',
    options: ['正常', '大便溏稀', '便秘', '大便干结', '大便粘马桶', '其他'],
  },
  {
    id: 5,
    name: '饮食',
    icon: '05',
    options: ['正常', '食欲不振', '食欲亢进', '厌食', '恶心', '呕吐', '口味偏重', '口味偏淡', '其他'],
  },
  {
    id: 6,
    name: '胸腹',
    icon: '06',
    options: ['正常', '胸闷', '胸痛', '腹胀', '腹痛', '胃痛', '嗳气', '反酸', '其他'],
  },
  {
    id: 7,
    name: '耳目',
    icon: '07',
    options: ['正常', '耳鸣', '听力下降', '视力模糊', '眼花', '眼睛干涩', '其他'],
  },
  {
    id: 8,
    name: '口渴',
    icon: '08',
    options: ['正常', '口干', '口渴喜冷饮', '口渴喜热饮', '口苦', '口中粘腻', '无明显口渴', '其他'],
  },
  {
    id: 9,
    name: '睡眠',
    icon: '09',
    options: ['正常', '失眠', '多梦', '易醒', '嗜睡', '难以入睡', '其他'],
  },
  {
    id: 10,
    name: '舌脉',
    icon: '10',
    options: ['正常', '舌苔薄白', '舌苔黄', '舌苔厚腻', '舌质淡红', '舌质红', '脉浮', '脉沉', '脉滑', '脉数', '其他'],
  },
]

interface Selection {
  dimensionId: number
  value: string
  customInput?: string
}

export const TenInquiryOptions: React.FC<TenInquiryOptionsProps> = ({ onSelect, disabled }) => {
  const [expandedId, setExpandedId] = useState<number | null>(null)
  const [selections, setSelections] = useState<Map<number, Selection>>(new Map())
  const [customInputId, setCustomInputId] = useState<number | null>(null)
  const [customInputValue, setCustomInputValue] = useState('')

  const expandNextDimension = (dimensionId: number) => {
    const currentIndex = TEN_INQUIRY.findIndex((item) => item.id === dimensionId)
    const nextItem = TEN_INQUIRY[currentIndex + 1]
    setExpandedId(nextItem ? nextItem.id : null)
  }

  const toggleExpand = (id: number) => {
    if (disabled) return
    setExpandedId(expandedId === id ? null : id)
  }

  const selectOption = (dimensionId: number, option: string) => {
    if (disabled) return

    const newSelections = new Map(selections)

    if (option === '其他') {
      // Show custom input for this dimension
      setCustomInputId(dimensionId)
      setCustomInputValue('')
      // Keep the selection but mark as custom
      newSelections.set(dimensionId, { dimensionId, value: '其他' })
      setSelections(newSelections)
      setExpandedId(dimensionId)
    } else {
      // Normal selection - remove custom input if exists
      if (customInputId === dimensionId) {
        setCustomInputId(null)
        setCustomInputValue('')
      }
      newSelections.set(dimensionId, { dimensionId, value: option })
      setSelections(newSelections)
      expandNextDimension(dimensionId)
    }
  }

  const handleCustomInputChange = (value: string) => {
    setCustomInputValue(value)
    if (customInputId !== null) {
      const newSelections = new Map(selections)
      newSelections.set(customInputId, {
        dimensionId: customInputId,
        value: '其他',
        customInput: value,
      })
      setSelections(newSelections)
    }
  }

  const handleCustomInputKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && customInputId !== null && customInputValue.trim()) {
      event.preventDefault()
      expandNextDimension(customInputId)
    }
  }

  const handleConfirm = () => {
    const result = Array.from(selections.values())
      .filter(s => s.value && s.value !== '正常' && (s.value !== '其他' || s.customInput?.trim()))
      .map(s => ({
        dimensionId: s.dimensionId,
        value: s.value === '其他' ? s.customInput!.trim() : s.value,
      }))
    if (result.length > 0) {
      onSelect(result)
    }
  }

  const hasValidSelection = Array.from(selections.values()).some(s => {
    if (s.value === '其他') {
      return !!s.customInput
    }
    return s.value && s.value !== '正常'
  })

  return (
    <div className="ten-inquiry-container">
      <div className="ten-inquiry-title">请补充或修正相关症状</div>
      <div className="ten-inquiry-list">
        {TEN_INQUIRY.map((item) => {
          const isExpanded = expandedId === item.id
          const selection = selections.get(item.id)
          const selectedValue = selection?.value
          const showCustomInput = customInputId === item.id && selectedValue === '其他'

          return (
            <div key={item.id} className={`ten-inquiry-item ${selectedValue && selectedValue !== '正常' ? 'has-selection' : ''}`}>
              <div className="ten-inquiry-header" onClick={() => toggleExpand(item.id)}>
                <div className="ten-inquiry-main">
                  <span className="ten-inquiry-icon">{item.icon}</span>
                  <span className="ten-inquiry-name">{item.name}</span>
                </div>
                <div className="ten-inquiry-actions">
                  {selection && selectedValue === '正常' && (
                    <span className="ten-inquiry-selected-value normal">正常</span>
                  )}
                  {selection && selectedValue && selectedValue !== '正常' && (
                    <span className="ten-inquiry-selected-value">{selection.customInput || selectedValue}</span>
                  )}
                  <span className={`ten-inquiry-arrow ${isExpanded ? 'expanded' : ''}`}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="6 9 12 15 18 9" />
                    </svg>
                  </span>
                </div>
              </div>
              {isExpanded && (
                <div className="ten-inquiry-options">
                  {item.options.map((option) => (
                    <button
                      key={option}
                      className={`ten-inquiry-option ${selectedValue === option ? 'selected' : ''}`}
                      onClick={() => selectOption(item.id, option)}
                      disabled={disabled}
                    >
                      {option}
                    </button>
                  ))}
                </div>
              )}
              {showCustomInput && (
                <div className="ten-inquiry-custom-input">
                  <input
                    type="text"
                    className="custom-input-field"
                    placeholder="请输入您的症状..."
                    value={customInputValue}
                    onChange={(e) => handleCustomInputChange(e.target.value)}
                    onKeyDown={handleCustomInputKeyDown}
                    autoFocus
                    disabled={disabled}
                  />
                </div>
              )}
            </div>
          )
        })}
      </div>
      <div className="ten-inquiry-footer">
        <div className="ten-inquiry-hint">
          {hasValidSelection
            ? `已选择 ${Array.from(selections.values()).filter(s => s.value && s.value !== '正常' && (s.value !== '其他' || s.customInput)).length} 项`
            : '请选择您有的症状'}
        </div>
        <button
          className="ten-inquiry-confirm"
          onClick={handleConfirm}
          disabled={!hasValidSelection || disabled}
        >
          确认选择
        </button>
      </div>
    </div>
  )
}

export const parseTenInquiryOptions = (content: string): number[] | null => {
  const match = content.match(/[\d,，\s]+/)
  if (!match) return null
  const numbers = match[0].match(/\d+/g)
  if (!numbers) return null
  return numbers.map(Number).filter((n) => n >= 1 && n <= 10)
}
