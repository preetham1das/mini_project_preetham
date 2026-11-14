import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import './App.css'

class AppErrorBoundary extends React.Component {
  state = { hasError: false, error: null }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, errorInfo) {
    console.error('Application render failed:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return (
        <main role="alert" style={{
          minHeight: '100vh',
          display: 'grid',
          placeContent: 'center',
          gap: 12,
          padding: 24,
          color: '#e6edf7',
          background: '#0b0f17',
          fontFamily: 'system-ui, sans-serif',
        }}>
          <h1>ForestWatch could not display this page</h1>
          <p>Reload the page. If the problem continues, check the browser console for the render error.</p>
          {this.state.error?.message && <pre style={{ whiteSpace: 'pre-wrap' }}>{this.state.error.message}</pre>}
          <button type="button" onClick={() => window.location.reload()}>Reload application</button>
        </main>
      )
    }

    return this.props.children
  }
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AppErrorBoundary>
      <App />
    </AppErrorBoundary>
  </React.StrictMode>,
)
