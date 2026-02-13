// frontend/src/components/ChatInterface.jsx
import React, { useState, useEffect, useRef } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { firstAidAPI } from '../services/api'

function ChatInterface({ conversationId, onToggleHistory }) {
  const { user, isGuest, logout } = useAuth()
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [currentConvId, setCurrentConvId] = useState(conversationId)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    if (conversationId && !isGuest) {
      loadConversation(conversationId)
      setCurrentConvId(conversationId)
    } else {
      setMessages([
        {
          type: 'system',
          content: isGuest
            ? 'Welcome. Please describe your first aid situation or query.'
            : `Welcome, ${
                user?.full_name || user?.username
              }. How may I assist you today?`,
          timestamp: new Date().toISOString(),
        },
      ])
    }
  }, [conversationId, user, isGuest])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const loadConversation = async (convId) => {
    try {
      const response = await firstAidAPI.getConversation(convId)
      const loadedMessages = response.messages.map((msg) => ({
        type: msg.role === 'user' ? 'user' : 'assistant',
        content: msg.content,
        sources: msg.sources,
        confidence_score: msg.confidence_score,
        timestamp: msg.timestamp,
      }))
      setMessages(loadedMessages)
    } catch (error) {
      console.error('Failed to load conversation:', error)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    await sendQuery(input)
  }

  const sendQuery = async (queryText) => {
    if (!queryText.trim() || loading) return

    const userMessage = {
      type: 'user',
      content: queryText,
      timestamp: new Date().toISOString(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      const response = await firstAidAPI.sendQuery(
        queryText,
        isGuest ? null : currentConvId
      )

      const assistantMessage = {
        type: 'assistant',
        content: response.response,
        sources: response.sources,
        confidence_score: response.confidence_score,
        timestamp: response.timestamp,
      }

      setMessages((prev) => [...prev, assistantMessage])

      if (!isGuest && response.conversation_id) {
        setCurrentConvId(response.conversation_id)
      }
    } catch (error) {
      console.error('Query error:', error)
      setMessages((prev) => [
        ...prev,
        {
          type: 'error',
          content:
            'An error occurred. For life-threatening emergencies, call 100/108 immediately.',
          timestamp: new Date().toISOString(),
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleNewChat = () => {
    setCurrentConvId(null)
    setMessages([
      {
        type: 'system',
        content: 'New consultation started. How may I assist you?',
        timestamp: new Date().toISOString(),
      },
    ])
  }

  const getConfidenceColor = (score) => {
    if (score >= 80) return 'bg-emerald-100 text-emerald-800 border-emerald-200'
    if (score >= 65) return 'bg-amber-100 text-amber-800 border-amber-200'
    return 'bg-red-100 text-red-800 border-red-200'
  }

  const getConfidenceLabel = (score) => {
    if (score >= 80) return 'High Confidence'
    if (score >= 65) return 'Moderate Confidence'
    return 'Low Confidence'
  }

  const renderMessage = (message, index) => {
    switch (message.type) {
      case 'user':
        return (
          <div
            key={index}
            className="mb-6 flex flex-col items-end animate-fadeIn"
          >
            <div className="bg-slate-700 text-white px-5 py-3.5 rounded-lg rounded-br-sm max-w-[75%] shadow-sm">
              <p className="whitespace-pre-wrap leading-relaxed">
                {message.content}
              </p>
            </div>
            <span className="text-xs text-slate-500 mt-1.5">
              {new Date(message.timestamp).toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit',
              })}
            </span>
          </div>
        )

      case 'assistant':
        return (
          <div
            key={index}
            className="mb-6 flex flex-col items-start animate-fadeIn"
          >
            <div className="bg-white px-6 py-4 rounded-lg rounded-bl-sm max-w-[85%] shadow-sm border border-slate-200">
              <div className="flex justify-between items-center mb-3 pb-3 border-b border-slate-100">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 bg-slate-700 rounded-full flex items-center justify-center text-white text-sm font-semibold">
                    FA
                  </div>
                  <span className="text-slate-700 font-semibold text-sm">
                    First Aid Assistant
                  </span>
                </div>
                {message.confidence_score !== undefined && (
                  <span
                    className={`px-3 py-1 rounded-md text-xs font-medium border ${getConfidenceColor(
                      message.confidence_score
                    )}`}
                  >
                    {getConfidenceLabel(message.confidence_score)} (
                    {message.confidence_score.toFixed(0)}%)
                  </span>
                )}
              </div>

              <div className="text-slate-800 leading-relaxed whitespace-pre-wrap">
                {message.content}
              </div>
            </div>
            <span className="text-xs text-slate-500 mt-1.5">
              {new Date(message.timestamp).toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit',
              })}
            </span>
          </div>
        )

      case 'system':
        return (
          <div
            key={index}
            className="mb-6 text-center text-slate-600 p-4 bg-slate-50 rounded-lg mx-auto max-w-2xl border border-slate-200"
          >
            <p className="text-sm">{message.content}</p>
          </div>
        )

      case 'error':
        return (
          <div
            key={index}
            className="mb-6 bg-red-50 text-red-800 p-4 rounded-lg border border-red-200 max-w-2xl mx-auto"
          >
            <div className="flex items-start gap-3">
              <span className="text-xl">⚠️</span>
              <p className="text-sm leading-relaxed">{message.content}</p>
            </div>
          </div>
        )

      default:
        return null
    }
  }

  const suggestedQuestions = [
    'Management of severe bleeding',
    'Treatment protocol for second-degree burns',
    'CPR procedure for adults',
    'Choking emergency response',
    'Sprained ankle treatment',
    'Recognition of shock symptoms',
  ]

  return (
    <div className="flex flex-col h-screen bg-slate-50">
      {/* Header */}
      <div className="bg-slate-800 text-white px-6 py-4 flex items-center justify-between shadow-md border-b border-slate-700">
        {!isGuest && (
          <button
            onClick={onToggleHistory}
            className="text-xl px-3 py-2 hover:bg-slate-700 rounded-md transition-colors"
            title="Consultation History"
          >
            ☰
          </button>
        )}
        <div className="flex-1 mx-4">
          <h1 className="text-xl font-semibold tracking-tight">
             First Aid Assistant
          </h1>
          <p className="text-sm text-slate-300 mt-0.5">
            {isGuest
              ? 'Guest Session - Not Saved'
              : `Logged in as ${user?.username}`}
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={handleNewChat}
            className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-md font-medium text-sm transition-colors"
          >
            New Consultation
          </button>
          <button
            onClick={logout}
            className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-md font-medium text-sm transition-colors"
          >
            {isGuest ? 'Sign In' : 'Sign Out'}
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-8">
        {messages.map((message, index) => renderMessage(message, index))}

        {loading && (
          <div className="flex items-center gap-3 p-4 bg-white rounded-lg shadow-sm border border-slate-200 animate-fadeIn max-w-md">
            <div className="flex gap-1.5">
              <span className="w-2 h-2 bg-slate-600 rounded-full animate-bounce"></span>
              <span
                className="w-2 h-2 bg-slate-600 rounded-full animate-bounce"
                style={{ animationDelay: '0.2s' }}
              ></span>
              <span
                className="w-2 h-2 bg-slate-600 rounded-full animate-bounce"
                style={{ animationDelay: '0.4s' }}
              ></span>
            </div>
            <p className="text-slate-600 text-sm">
              Analyzing medical protocols...
            </p>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Suggestions */}
      {messages.length <= 1 && !loading && (
        <div className="px-8 py-6 bg-white border-t border-slate-200">
          <p className="font-semibold text-slate-700 mb-4 text-sm">
            Common First Aid Inquiries:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {suggestedQuestions.map((q, idx) => (
              <button
                key={idx}
                onClick={() => sendQuery(q)}
                className="px-4 py-3 bg-slate-50 border border-slate-200 rounded-md text-left text-sm text-slate-700 transition-all hover:bg-slate-700 hover:text-white hover:border-slate-700 hover:shadow-sm"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Guest Mode Warning */}
      {isGuest && messages.length > 1 && (
        <div className="px-6 py-3 bg-amber-50 border-t border-amber-200">
          <p className="text-sm text-amber-900 text-center">
            <span className="font-semibold">Guest Mode:</span> Conversation will
            not be saved.{' '}
            <button
              onClick={logout}
              className="underline font-semibold hover:text-amber-800"
            >
              Sign in to save
            </button>
          </p>
        </div>
      )}

      {/* Input */}
      <form
        onSubmit={handleSubmit}
        className="px-6 py-5 bg-white border-t border-slate-200"
      >
        <div className="flex gap-3 items-center max-w-5xl mx-auto">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Describe your first aid query or emergency situation..."
            disabled={loading}
            className="flex-1 px-5 py-3.5 border border-slate-300 rounded-lg text-base focus:outline-none focus:border-slate-600 focus:ring-2 focus:ring-slate-600/20 disabled:bg-slate-100 disabled:cursor-not-allowed transition-all"
          />

          <button
            type="submit"
            disabled={!input.trim() || loading}
            className="px-6 py-3.5 bg-slate-700 text-white rounded-lg font-medium text-base hover:bg-slate-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-slate-700"
          >
            {loading ? 'Analyzing...' : 'Send'}
          </button>
        </div>

        <p className="text-center text-xs text-slate-500 mt-4 max-w-3xl mx-auto">
          <span className="font-semibold text-red-600">
            ⚠️ EMERGENCY DISCLAIMER:
          </span>{' '}
          This assistant provides informational guidance only. For
          life-threatening emergencies, call 100/108 immediately. Always seek
          professional medical care when needed.
        </p>
      </form>

      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fadeIn {
          animation: fadeIn 0.3s ease;
        }
      `}</style>
    </div>
  )
}

export default ChatInterface
