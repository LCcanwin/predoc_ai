import { useCallback } from 'react'

const authHeaders = (token?: string | null) => ({
  'Content-Type': 'application/json',
  ...(token ? { Authorization: `Bearer ${token}` } : {}),
})

interface StreamCallbacks {
  onThinking?: (status: string, message?: string) => void
  onMessage?: (content: string, isComplete?: boolean) => void
  onComplete?: (isComplete: boolean) => void
  onError?: (error: Error) => void
}

export const useChatStream = (token?: string | null) => {
  const startConsultation = useCallback(async (): Promise<string> => {
    try {
      const response = await fetch('/api/consultation/start', {
        method: 'POST',
        headers: authHeaders(token),
        body: JSON.stringify({}),
      })

      if (!response.ok) {
        throw new Error('Failed to start consultation')
      }

      const data = await response.json()
      return data.thread_id
    } catch (error) {
      console.error('Error starting consultation:', error)
      throw error
    }
  }, [token])

  const sendMessage = useCallback(async (
    threadId: string,
    content: string,
    callbacks: StreamCallbacks = {}
  ) => {
    const { onThinking, onMessage, onComplete, onError } = callbacks

    try {
      onThinking?.('thinking', '正在思考...')

      const response = await fetch(`/api/consultation/${threadId}/message`, {
        method: 'POST',
        headers: authHeaders(token),
        body: JSON.stringify({ content }),
      })

      if (!response.ok) {
        throw new Error('Failed to send message')
      }

      // Read the entire response as text
      const text = await response.text()
      console.log('Received response, length:', text.length)

      // Parse SSE events manually
      const events: { type: string; data: any }[] = []
      const eventBlocks = text.trim().split(/\r?\n\r?\n/)

      for (const block of eventBlocks) {
        const lines = block.split(/\r?\n/)
        let eventType = ''
        let eventData = ''

        for (const line of lines) {
          if (line.startsWith('event:')) {
            eventType = line.substring(6).trim()
          } else if (line.startsWith('data:')) {
            eventData = line.substring(5).trim()
          }
        }

        if (eventType && eventData) {
          try {
            const data = JSON.parse(eventData)
            events.push({ type: eventType, data })
            console.log('Parsed event:', eventType, data)
          } catch (e) {
            console.error('Failed to parse event data:', e)
          }
        }
      }

      // Process events
      let isComplete = false
      for (const event of events) {
        if (event.type === 'thinking') {
          onThinking?.('thinking', event.data.message || '正在思考...')
        } else if (event.type === 'message') {
          isComplete = event.data.is_complete || false
          onMessage?.(event.data.content, isComplete)
        } else if (event.type === 'complete') {
          isComplete = event.data.is_complete || isComplete
          onComplete?.(isComplete)
        } else if (event.type === 'error') {
          console.error('Server error:', event.data.message)
          onError?.(new Error(event.data.message))
        }
      }

    } catch (error) {
      console.error('Send message error:', error)
      onError?.(error as Error)
    }
  }, [token])

  return {
    startConsultation,
    sendMessage,
  }
}
