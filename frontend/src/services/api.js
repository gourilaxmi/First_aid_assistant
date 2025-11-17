import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 3000000,
})

// Add auth token to requests
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Handle token refresh on 401
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token')
      localStorage.removeItem('refresh_token')
      window.location.reload()
    }
    return Promise.reject(error)
  }
)

// API Service
const api = {
  // Health check
  async checkHealth() {
    const response = await apiClient.get('/health')
    return response.data
  },

  // Auth Endpoints
  async register(userData) {
    try {
      const response = await apiClient.post('/auth/register', userData)
      return response.data
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Registration failed')
    }
  },

  async login(username, password) {
    try {
      const formData = new FormData()
      formData.append('username', username)
      formData.append('password', password)

      const response = await apiClient.post('/auth/login', formData, {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      })
      return response.data
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Login failed')
    }
  },

  async getCurrentUser() {
    try {
      const response = await apiClient.get('/auth/me')
      return response.data
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Failed to get user')
    }
  },

  async logout() {
    try {
      const refreshToken = localStorage.getItem('refresh_token')
      const response = await apiClient.post('/auth/logout', {
        refresh_token: refreshToken,
      })
      return response.data
    } catch (error) {
      console.error('Logout error:', error)
      // Don't throw - logout should always succeed locally
    }
  },
}

// First Aid API
export const firstAidAPI = {
  // Query Endpoints
  async sendQuery(query, conversationId = null, topK = 10, minScore = 0.6) {
    try {
      const response = await apiClient.post('/api/query', {
        query,
        conversation_id: conversationId,
        top_k: topK,
        min_score: minScore,
      })
      return response.data
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Query failed')
    }
  },

  // Conversation Endpoints
  async getConversations(limit = 20) {
    try {
      const response = await apiClient.get('/api/conversations', {
        params: { limit },
      })
      return response.data
    } catch (error) {
      throw new Error(
        error.response?.data?.detail || 'Failed to get conversations'
      )
    }
  },

  async getConversation(conversationId) {
    try {
      const response = await apiClient.get(`/api/history/${conversationId}`)
      return response.data
    } catch (error) {
      throw new Error(
        error.response?.data?.detail || 'Failed to get conversation'
      )
    }
  },

  async updateConversationTitle(conversationId, title) {
    try {
      const response = await apiClient.put(
        `/api/conversations/${conversationId}/title`,
        null,
        {
          params: { title },
        }
      )
      return response.data
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Failed to update title')
    }
  },

  async deleteConversation(conversationId) {
    try {
      const response = await apiClient.delete(
        `/api/conversations/${conversationId}`
      )
      return response.data
    } catch (error) {
      throw new Error(
        error.response?.data?.detail || 'Failed to delete conversation'
      )
    }
  },

  // Performance Metrics
  async getMetrics() {
    try {
      const response = await apiClient.get('/api/metrics')
      return response.data
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Failed to get metrics')
    }
  },

  async resetMetrics() {
    try {
      const response = await apiClient.post('/api/metrics/reset')
      return response.data
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Failed to reset metrics')
    }
  },
}

export default api
