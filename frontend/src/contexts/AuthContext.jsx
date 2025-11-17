import React, { createContext, useState, useEffect, useContext } from 'react'
import api from '../services/api'

const AuthContext = createContext(null)

function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [token, setToken] = useState(null)
  const [loading, setLoading] = useState(true)
  const [isGuest, setIsGuest] = useState(false)

  useEffect(() => {
    const storedToken = localStorage.getItem('access_token')
    const guestMode = localStorage.getItem('guest_mode')

    if (guestMode === 'true') {
      setIsGuest(true)
      setLoading(false)
    } else if (storedToken) {
      fetchCurrentUser(storedToken)
    } else {
      setLoading(false)
    }
  }, [])

  const fetchCurrentUser = async (accessToken) => {
    try {
      const userData = await api.getCurrentUser(accessToken)
      setUser(userData)
      setToken(accessToken)
    } catch (error) {
      console.error('Failed to fetch user:', error)
      logout()
    } finally {
      setLoading(false)
    }
  }

  const login = async (username, password) => {
    try {
      const response = await api.login(username, password)
      localStorage.setItem('access_token', response.access_token)
      localStorage.setItem('refresh_token', response.refresh_token)
      localStorage.removeItem('guest_mode')
      setToken(response.access_token)
      setUser(response.user)
      setIsGuest(false)
      return { success: true }
    } catch (error) {
      return { success: false, error: error.message }
    }
  }

  const register = async (userData) => {
    try {
      await api.register(userData)
      return { success: true }
    } catch (error) {
      return { success: false, error: error.message }
    }
  }

  const logout = async () => {
    try {
      const refreshToken = localStorage.getItem('refresh_token')
      if (token && refreshToken) {
        await api.logout(token, refreshToken)
      }
    } catch (error) {
      console.error('Logout error:', error)
    } finally {
      localStorage.removeItem('access_token')
      localStorage.removeItem('refresh_token')
      localStorage.removeItem('guest_mode')
      setUser(null)
      setToken(null)
      setIsGuest(false)
    }
  }

  const continueAsGuest = () => {
    localStorage.setItem('guest_mode', 'true')
    setIsGuest(true)
    setLoading(false)
    return { success: true }
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        loading,
        isGuest,
        isAuthenticated: !!(token || isGuest),
        login,
        register,
        logout,
        continueAsGuest,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

function useAuth() {
  const context = useContext(AuthContext)
  if (!context) throw new Error('useAuth must be used within AuthProvider')
  return context
}

export { AuthProvider, useAuth }
