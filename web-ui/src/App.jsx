import { useState } from 'react'
import './index.css'

const EXAMPLES = [
  {
    name: "Machine Learning (CS + Stats)",
    text: "We propose a novel deep convolutional neural network architecture for multi-label image classification. By leveraging self-attention mechanisms and residual connections, our model achieves state-of-the-art performance on several benchmark datasets. We also perform a rigorous statistical analysis of the training dynamics, proving convergence under mild assumptions."
  },
  {
    name: "Quantum Topology (Physics + Math)",
    text: "This paper analyzes the topological properties of quantum spin systems in three dimensions. We derive a set of exact mathematical formulations for the ground state degeneracy using algebraic topology. We show that the Hamiltonian exhibits topological order, which remains stable under arbitrary local perturbations."
  },
  {
    name: "Option Pricing (Quant Finance)",
    text: "We formulate a new stochastic volatility model for pricing exotic options in incomplete markets. Using a system of partial differential equations, we find semi-analytical solutions for European call options. Empirical calibration to S&P 500 options shows that our model resolves the volatility smile anomaly more effectively than Black-Scholes."
  }
]

function App() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [metadata, setMetadata] = useState(null)
  
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
    setMetadata(null)
    
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
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to classify text. Is the Python server running?')
      }

      const data = await response.json()
      setPredictions(data.predictions)
      if (data.metadata) {
        setMetadata(data.metadata)
      }
      
    } catch (err) {
      setError(err.message || 'An error occurred during classification.')
    } finally {
      setLoading(false)
    }
  }

  const handleExampleClick = (exampleText) => {
    setText(exampleText)
    setError('')
    setMetadata(null)
    const reset = {}
    for (let key in predictions) reset[key] = false
    setPredictions(reset)
  }

  const activeCount = Object.values(predictions).filter(Boolean).length

  return (
    <div className="app-container">
      <div className="header">
        <h1>Research Paper Classifier</h1>
        <p>Multi-Label NLP Abstract Categorization</p>
      </div>

      <div className="examples-section">
        <span className="examples-label">Quick Test Examples:</span>
        <div className="examples-list">
          {EXAMPLES.map((ex, idx) => (
            <button 
              key={idx} 
              className="btn-example" 
              onClick={() => handleExampleClick(ex.text)}
              disabled={loading}
            >
              {ex.name}
            </button>
          ))}
        </div>
      </div>

      <div className="input-container">
        <textarea 
          placeholder="Paste Research Paper Abstract or Content here (minimum 10 characters)..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={loading}
        />
        <div className="textarea-footer">
          <span>{text.length} characters</span>
          <span>{text.split(/\s+/).filter(Boolean).length} words</span>
        </div>
      </div>

      {error && <div className="error-msg">{error}</div>}

      <button 
        className="btn-classify" 
        onClick={handleClassify}
        disabled={loading || text.trim().length < 10}
      >
        {loading ? <span className="loader"></span> : 'Classify Document'}
      </button>

      <div className="results-container">
        <div className="results-header">
          <h3>Predicted Domains {activeCount > 0 ? `(${activeCount} matched)` : ''}</h3>
        </div>
        <div className="results-grid">
          {Object.entries(predictions).map(([subject, active]) => (
            <div 
              key={subject} 
              className={`badge ${active ? 'active' : ''}`}
            >
              <span className="badge-dot"></span>
              {subject}
            </div>
          ))}
        </div>
      </div>

      {metadata && (
        <div className="metadata-container">
          <div className="metadata-item">
            <span className="meta-label">Inference Latency:</span>
            <span className="meta-value">{metadata.latency_ms} ms</span>
          </div>
          <div className="metadata-divider">|</div>
          <div className="metadata-item">
            <span className="meta-label">Word Count:</span>
            <span className="meta-value">{metadata.word_count} words</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
