// frontend/src/components/ConversationHistory.jsx
import React, { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { firstAidAPI } from '../services/api'

function ConversationHistory({ onSelect, onClose, selectedId }) {
  const { token } = useAuth()
  const [conversations, setConversations] = useState([])
  const [loading, setLoading] = useState(true)
  const [editingId, setEditingId] = useState(null)
  const [editTitle, setEditTitle] = useState('')

  useEffect(() => {
    loadConversations()
  }, [])

  const loadConversations = async () => {
    try {
      const response = await firstAidAPI.getConversations()
      setConversations(response.conversations || [])
    } catch (error) {
      console.error('Failed to load conversations:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (convId, e) => {
    e.stopPropagation()
    if (!confirm('Delete this consultation record?')) return

    try {
      await firstAidAPI.deleteConversation(convId)
      setConversations((prev) =>
        prev.filter((c) => c.conversation_id !== convId)
      )
    } catch (error) {
      console.error('Failed to delete:', error)
      alert('Failed to delete consultation')
    }
  }

  const handleEdit = (conv, e) => {
    e.stopPropagation()
    setEditingId(conv.conversation_id)
    setEditTitle(conv.title)
  }

  const handleSaveTitle = async (convId, e) => {
    e.stopPropagation()

    try {
      await firstAidAPI.updateConversationTitle(convId, editTitle)
      setConversations((prev) =>
        prev.map((c) =>
          c.conversation_id === convId ? { ...c, title: editTitle } : c
        )
      )
      setEditingId(null)
    } catch (error) {
      console.error('Failed to update title:', error)
      alert('Failed to update title')
    }
  }

  const formatDate = (dateString) => {
    const date = new Date(dateString)
    const now = new Date()
    const diffMs = now - date
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    const diffDays = Math.floor(diffMs / 86400000)

    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays < 7) return `${diffDays}d ago`
    return date.toLocaleDateString()
  }

  if (loading) {
    return (
      <div className="h-full flex flex-col bg-white border-r border-slate-200">
        <div className="p-6 bg-slate-800 text-white flex justify-between items-center border-b border-slate-700">
          <h2 className="text-lg font-semibold">Consultation History</h2>
          <button
            onClick={onClose}
            className="w-9 h-9 flex items-center justify-center hover:bg-slate-700 rounded-md text-xl transition-colors"
          >
            ×
          </button>
        </div>
        <div className="flex items-center justify-center p-8 text-slate-500">
          <div className="text-center">
            <div className="inline-block w-8 h-8 border-4 border-slate-300 border-t-slate-600 rounded-full animate-spin mb-3"></div>
            <p className="text-sm">Loading consultations...</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col bg-slate-50 border-r border-slate-200">
      <div className="p-6 bg-slate-800 text-white flex justify-between items-center border-b border-slate-700">
        <h2 className="text-lg font-semibold">Consultation History</h2>
        <button
          onClick={onClose}
          className="w-9 h-9 flex items-center justify-center hover:bg-slate-700 rounded-md text-xl transition-colors"
        >
          ×
        </button>
      </div>

      {conversations.length === 0 ? (
        <div className="text-center p-12 text-slate-500">
          <div className="text-4xl mb-4">📋</div>
          <p className="font-medium mb-2">No Consultation Records</p>
          <p className="text-sm">
            Start a new consultation to see your history here
          </p>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto p-4">
          {conversations.map((conv) => (
            <div
              key={conv.conversation_id}
              className={`p-4 border rounded-lg mb-3 cursor-pointer transition-all bg-white ${
                selectedId === conv.conversation_id
                  ? 'border-slate-700 shadow-md ring-2 ring-slate-700/10'
                  : 'border-slate-200 hover:border-slate-400 hover:shadow-sm'
              }`}
              onClick={() => onSelect(conv.conversation_id)}
            >
              {editingId === conv.conversation_id ? (
                <div
                  className="flex gap-2 items-center"
                  onClick={(e) => e.stopPropagation()}
                >
                  <input
                    type="text"
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    className="flex-1 px-3 py-2 border border-slate-300 rounded-md text-sm focus:outline-none focus:border-slate-600 focus:ring-2 focus:ring-slate-600/20"
                    autoFocus
                  />
                  <button
                    onClick={(e) => handleSaveTitle(conv.conversation_id, e)}
                    className="px-3 py-2 bg-emerald-600 text-white rounded-md hover:bg-emerald-700 text-sm font-medium transition-colors"
                  >
                    Save
                  </button>
                  <button
                    onClick={() => setEditingId(null)}
                    className="px-3 py-2 bg-slate-200 text-slate-700 rounded-md hover:bg-slate-300 text-sm font-medium transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <>
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="text-sm font-semibold text-slate-800 flex-1 pr-2 line-clamp-2">
                      {conv.title}
                    </h3>
                    <div className="flex gap-1">
                      <button
                        onClick={(e) => handleEdit(conv, e)}
                        className="p-1.5 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded transition-colors"
                        title="Edit title"
                      >
                        <svg
                          className="w-4 h-4"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                          />
                        </svg>
                      </button>
                      <button
                        onClick={(e) => handleDelete(conv.conversation_id, e)}
                        className="p-1.5 text-slate-500 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
                        title="Delete"
                      >
                        <svg
                          className="w-4 h-4"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                          />
                        </svg>
                      </button>
                    </div>
                  </div>
                  <div className="flex gap-4 text-xs text-slate-500 mb-2">
                    <span className="flex items-center gap-1">
                      <svg
                        className="w-3.5 h-3.5"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                        />
                      </svg>
                      {conv.message_count} messages
                    </span>
                    <span className="flex items-center gap-1">
                      <svg
                        className="w-3.5 h-3.5"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                      </svg>
                      {formatDate(conv.updated_at)}
                    </span>
                  </div>
                  {conv.last_query && (
                    <p className="text-sm text-slate-600 line-clamp-2 leading-relaxed">
                      {conv.last_query}
                    </p>
                  )}
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default ConversationHistory
