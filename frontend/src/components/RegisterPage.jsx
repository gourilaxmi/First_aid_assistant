import React, { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'

function RegisterPage({ onSwitch }) {
  const { register } = useAuth()
  const [formData, setFormData] = useState({
    email: '',
    username: '',
    password: '',
    confirmPassword: '',
    full_name: '',
  })
  const [error, setError] = useState('')
  const [success, setSuccess] = useState(false)
  const [loading, setLoading] = useState(false)

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')

    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match')
      return
    }

    if (formData.password.length < 8) {
      setError('Password must be at least 8 characters')
      return
    }

    setLoading(true)

    const result = await register({
      email: formData.email,
      username: formData.username,
      password: formData.password,
      full_name: formData.full_name,
    })

    if (result.success) {
      setSuccess(true)
      setTimeout(() => onSwitch(), 2000)
    } else {
      setError(result.error)
    }

    setLoading(false)
  }

  if (success) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-800 p-5">
        <div className="bg-white p-10 rounded-xl shadow-2xl max-w-md w-full border border-slate-200">
          <div className="text-center p-10">
            <div className="w-20 h-20 bg-emerald-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg
                className="w-12 h-12 text-emerald-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M5 13l4 4L19 7"
                />
              </svg>
            </div>
            <h2 className="text-emerald-600 mb-3 text-2xl font-bold">
              Registration Successful
            </h2>
            <p className="text-slate-600">Redirecting to sign in...</p>
          </div>
        </div>
      </div>
    )
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
                d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z"
              />
            </svg>
          </div>
          <h1 className="text-slate-800 mb-2 text-2xl font-bold">
            Create Account
          </h1>
          <p className="text-slate-600 text-sm">
            Join the First Aid Assistant Platform
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

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <label className="font-semibold text-slate-700 text-sm">
              Email Address *
            </label>
            <input
              type="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              required
              disabled={loading}
              className="px-4 py-3 border border-slate-300 rounded-lg text-base transition-all focus:outline-none focus:border-slate-600 focus:ring-2 focus:ring-slate-600/20 disabled:bg-slate-100"
            />
          </div>

          <div className="flex flex-col gap-2">
            <label className="font-semibold text-slate-700 text-sm">
              Username *
            </label>
            <input
              type="text"
              name="username"
              value={formData.username}
              onChange={handleChange}
              required
              minLength={3}
              disabled={loading}
              className="px-4 py-3 border border-slate-300 rounded-lg text-base transition-all focus:outline-none focus:border-slate-600 focus:ring-2 focus:ring-slate-600/20 disabled:bg-slate-100"
            />
          </div>

          <div className="flex flex-col gap-2">
            <label className="font-semibold text-slate-700 text-sm">
              Full Name
            </label>
            <input
              type="text"
              name="full_name"
              value={formData.full_name}
              onChange={handleChange}
              disabled={loading}
              placeholder="Optional"
              className="px-4 py-3 border border-slate-300 rounded-lg text-base transition-all focus:outline-none focus:border-slate-600 focus:ring-2 focus:ring-slate-600/20 disabled:bg-slate-100"
            />
          </div>

          <div className="flex flex-col gap-2">
            <label className="font-semibold text-slate-700 text-sm">
              Password *
            </label>
            <input
              type="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              required
              minLength={8}
              disabled={loading}
              className="px-4 py-3 border border-slate-300 rounded-lg text-base transition-all focus:outline-none focus:border-slate-600 focus:ring-2 focus:ring-slate-600/20 disabled:bg-slate-100"
            />
            <p className="text-xs text-slate-500">
              Minimum 8 characters required
            </p>
          </div>

          <div className="flex flex-col gap-2">
            <label className="font-semibold text-slate-700 text-sm">
              Confirm Password *
            </label>
            <input
              type="password"
              name="confirmPassword"
              value={formData.confirmPassword}
              onChange={handleChange}
              required
              disabled={loading}
              className="px-4 py-3 border border-slate-300 rounded-lg text-base transition-all focus:outline-none focus:border-slate-600 focus:ring-2 focus:ring-slate-600/20 disabled:bg-slate-100"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="py-3.5 bg-slate-700 text-white rounded-lg text-base font-semibold transition-all hover:bg-slate-600 hover:shadow-lg disabled:opacity-60 disabled:hover:bg-slate-700 mt-2"
          >
            {loading ? 'Creating Account...' : 'Create Account'}
          </button>
        </form>

        <div className="mt-5 text-center text-slate-600 text-sm">
          <p>
            Already have an account?{' '}
            <button
              onClick={onSwitch}
              className="text-slate-700 font-semibold underline hover:text-slate-900"
            >
              Sign In
            </button>
          </p>
        </div>

        <div className="mt-6 pt-5 border-t border-slate-200">
          <p className="text-xs text-slate-500 text-center">
            By creating an account, you agree to use this platform for
            informational purposes only
          </p>
        </div>
      </div>
    </div>
  )
}

export default RegisterPage
