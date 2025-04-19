import React, { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { useDispatch } from 'react-redux'
import { login } from './store/authSlice'
import authService from './appwrite/appwrite'

// Pages
import Home from './pages/Home'
import Login from './pages/Login'
import Register from './pages/Register'
import Dashboard from './pages/Dashboard'
import PDFView from './pages/PDFView'

// Components
import ProtectedRoute from './components/ProtectedRoute'

function App() {
  const [loading, setLoading] = useState(true)
  const dispatch = useDispatch()

  useEffect(() => {
    // Check if user is already logged in
    authService.getCurrentUser()
      .then((userData) => {
        if (userData) {
          dispatch(login({ userData }))
        }
      })
      .finally(() => setLoading(false))
  }, [dispatch])

  return loading ? (
    <div className="flex justify-center items-center min-h-screen">
      Loading...
    </div>
  ) : (
    <Router>
      <Routes>
        <Route 
          path="/" 
          element={
            <Home />
          } 
        />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route 
          path="/dashboard" 
          element={
            <ProtectedRoute authentication={true}>
              <Dashboard />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/pdf/:id" 
          element={
            <ProtectedRoute authentication={true}>
              <PDFView />
            </ProtectedRoute>
          } 
        />
      </Routes>
    </Router>
  )
}

export default App