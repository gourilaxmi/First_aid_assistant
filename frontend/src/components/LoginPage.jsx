import React, { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'

function LoginPage({ onSwitch }) {
  const { login, continueAsGuest } = useAuth()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    const result = await login(username, password)
    if (!result.success) {
      setError(result.error)
    }
    setLoading(false)
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-800 p-5">
      <div className="bg-white p-10 rounded-xl shadow-2xl max-w-md w-full border border-slate-200">
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-slate-700 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg
              className="w-10 h-10 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
              />
            </svg>
          </div>
          <h1 className="text-slate-800 mb-2 text-2xl font-bold">
            First Aid Assistant
          </h1>
          <p className="text-slate-600 text-sm">
            Professional Medical Guidance Platform
          </p>
        </div>

        {error && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 mb-5 text-sm flex items-start gap-2">
            <svg
              className="w-5 h-5 shrink-0 mt-0.5"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                clipRule="evenodd"
              />
            </svg>
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="flex flex-col gap-5">
          <div className="flex flex-col gap-2">
            <label className="font-semibold text-slate-700 text-sm">
              Username or Email
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              disabled={loading}
              className="px-4 py-3 border border-slate-300 rounded-lg text-base transition-all focus:outline-none focus:border-slate-600 focus:ring-2 focus:ring-slate-600/20 disabled:bg-slate-100"
            />
          </div>

          <div className="flex flex-col gap-2">
            <label className="font-semibold text-slate-700 text-sm">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              disabled={loading}
              className="px-4 py-3 border border-slate-300 rounded-lg text-base transition-all focus:outline-none focus:border-slate-600 focus:ring-2 focus:ring-slate-600/20 disabled:bg-slate-100"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="py-3.5 bg-slate-700 text-white rounded-lg text-base font-semibold transition-all hover:bg-slate-600 hover:shadow-lg disabled:opacity-60 disabled:hover:bg-slate-700"
          >
            {loading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>

        <div className="mt-6">
          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-slate-300"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-white text-slate-500">Or</span>
            </div>
          </div>

          <button
            onClick={continueAsGuest}
            className="w-full mt-4 py-3.5 bg-slate-100 text-slate-700 rounded-lg text-base font-semibold transition-all hover:bg-slate-200 border border-slate-200"
          >
            Continue as Guest
          </button>
        </div>

        <div className="mt-5 text-center text-slate-600 text-sm">
          <p>
            Don't have an account?{' '}
            <button
              onClick={onSwitch}
              className="text-slate-700 font-semibold underline hover:text-slate-900"
            >
              Create Account
            </button>
          </p>
        </div>

        <div className="mt-6 pt-5 border-t border-slate-200">
          <p className="text-xs text-slate-500 text-center">
            <span className="font-semibold text-red-600">⚠️</span> For
            life-threatening emergencies, call 108/100 immediately
          </p>
        </div>
      </div>
    </div>
  )
}

export default LoginPage
