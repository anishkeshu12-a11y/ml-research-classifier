import { useState } from 'react'
import './index.css'

function App() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  
  const [predictions, setPredictions] = useState({
    'Computer Science': false,
    'Physics': false,
    'Mathematics': false,
    'Statistics': false,
    'Quantitative Biology': false,
    'Quantitative Finance': false,
  })

  const handleClassify = async () => {
    if (!text.trim()) return

    setLoading(true)
    setError('')
    
    // Reset predictions
    const reset = {}
    for (let key in predictions) reset[key] = false
    setPredictions(reset)

    try {
      const response = await fetch('/api/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      })

      if (!response.ok) {
        throw new Error('Failed to classify text. Is the Python server running?')
      }

      const data = await response.json()
      setPredictions(data.predictions)
      
    } catch (err) {
      setError(err.message || 'An error occurred during classification.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <div className="header">
        <h1>Research Paper Classifier</h1>
        <p>AI-Powered Abstract Categorization</p>
      </div>

      <div className="input-container">
        <textarea 
          placeholder="Paste Research Paper Abstract or Content here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
      </div>

      {error && <div className="error-msg">{error}</div>}

      <button 
        className="btn-classify" 
        onClick={handleClassify}
        disabled={loading || !text.trim()}
      >
        {loading ? <span className="loader"></span> : 'Classify Document'}
      </button>

      <div className="results-grid">
        {Object.entries(predictions).map(([subject, active]) => (
          <div 
            key={subject} 
            className={`badge ${active ? 'active' : ''}`}
          >
            {subject}
          </div>
        ))}
      </div>
    </div>
  )
}

export default App
