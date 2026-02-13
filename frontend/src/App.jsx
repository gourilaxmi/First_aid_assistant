import React, { useState } from 'react'
import { AuthProvider, useAuth } from './contexts/AuthContext'
import LoginPage from './components/LoginPage'
import RegisterPage from './components/RegisterPage'
import ChatInterface from './components/ChatInterface'
import ConversationHistory from './components/ConversationHistory'

function AppContent() {
  const { user, isGuest, loading, isAuthenticated } = useAuth()
  const [showRegister, setShowRegister] = useState(false)
  const [showHistory, setShowHistory] = useState(false)
  const [selectedConversationId, setSelectedConversationId] = useState(null)

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-linear-to-br from-indigo-600 to-purple-700">
        <div className="text-white text-xl">Loading...</div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return showRegister ? (
      <RegisterPage onSwitch={() => setShowRegister(false)} />
    ) : (
      <LoginPage onSwitch={() => setShowRegister(true)} />
    )
  }

  return (
    <div className="flex h-screen">
      {/* Sidebar for logged-in users only */}
      {!isGuest && showHistory && (
        <div className="w-80 border-r border-gray-200">
          <ConversationHistory
            onSelect={(convId) => {
              setSelectedConversationId(convId)
              setShowHistory(false)
            }}
            onClose={() => setShowHistory(false)}
            selectedId={selectedConversationId}
          />
        </div>
      )}

      <div className="flex-1">
        <ChatInterface
          conversationId={selectedConversationId}
          onToggleHistory={() => setShowHistory(!showHistory)}
        />
      </div>
    </div>
  )
}

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  )
}

export default App
