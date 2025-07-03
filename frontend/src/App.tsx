import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  ScatterChart,
  Scatter
} from 'recharts';
import { buildApiUrl, config } from './config';
import './App.css';

interface ScoreHistory {
  step: number;
  score: number;
  timestamp: string;
  improvement?: number;
}

interface OptimizationStep {
  step: number;
  position: [number, number];
  direction: number;
  improvement: number;
  score: number;
  old_value?: number;
  new_value?: number;
}

interface MatrixHistory {
  step: number;
  matrix: number[][];
  timestamp: string;
}

interface CellHistory {
  step: number;
  value: number;
  position: string;
  improvement?: number;
}

interface PlotData {
  position: [number, number];
  current_value: number;
  values: number[];
  scores: number[];
  title: string;
}

interface StepPlotsData {
  step: number;
  plots_data: { [key: string]: PlotData };
  matrix_size: number;
  timestamp: string;
}

function App() {
  // State for matrices - back to text input
  const [HMatrixText, setHMatrixText] = useState<string>(
    '[[0, 1, 0],\n [1, 0, 1],\n [0, 1, 0]]'
  );
  
  const [GMatrixText, setGMatrixText] = useState<string>(
    '[[0.5, 0.3, 0.2],\n [0.3, 0.4, 0.1],\n [0.2, 0.1, 0.6]]'
  );

  // State for computation results
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string>('');
  const [isCalculating, setIsCalculating] = useState<boolean>(false);
  const [isOptimizing, setIsOptimizing] = useState<boolean>(false);
  
  // Optimization parameters
  const [stepSize, setStepSize] = useState<string>('0.01');
  const [numberOfSteps, setNumberOfSteps] = useState<string>('10');

  // Enhanced plotting state
  const [scoreHistory, setScoreHistory] = useState<ScoreHistory[]>([]);
  const [optimizationSteps, setOptimizationSteps] = useState<OptimizationStep[]>([]);
  const [matrixHistory, setMatrixHistory] = useState<MatrixHistory[]>([]);
  const [stepPlotsHistory, setStepPlotsHistory] = useState<StepPlotsData[]>([]);
  const [showPlots, setShowPlots] = useState<boolean>(true);

  // Use ref to control optimization stopping
  const optimizationShouldStop = useRef<boolean>(false);

  // Add step counter to ensure unique incrementing step numbers starting from 1
  const stepCounterRef = useRef(1);

  // Parse matrix from text
  const parseMatrix = (text: string): number[][] | null => {
    try {
      // Clean up the text and evaluate as JSON
      const cleanText = text.replace(/\s+/g, ' ').trim();
      const matrix = JSON.parse(cleanText);
      
      // Validate it's a proper 2D array
      if (!Array.isArray(matrix)) return null;
      if (!matrix.every(row => Array.isArray(row))) return null;
      if (!matrix.every(row => row.length === matrix[0].length)) return null;
      
      return matrix;
    } catch (e) {
      return null;
    }
  };

  // Get current matrices
  const getCurrentMatrices = () => {
    const H = parseMatrix(HMatrixText);
    const G = parseMatrix(GMatrixText);
    return { H, G };
  };

  // Load example data
  const loadExampleData = () => {
    setHMatrixText('[[0, 1, 0],\n [1, 0, 1],\n [0, 1, 0]]');
    setGMatrixText('[[0.5, 0.3, 0.2],\n [0.3, 0.4, 0.1],\n [0.2, 0.1, 0.6]]');
    setError('');
    setResults(null);
    clearAllHistory();
  };

  // Clear all history
  const clearAllHistory = () => {
    setScoreHistory([]);
    setOptimizationSteps([]);
    setMatrixHistory([]);
    setStepPlotsHistory([]);
    stepCounterRef.current = 1; // Reset step counter
  };

  // Add score to history
  const addScoreToHistory = (score: number, improvement?: number) => {
    const timestamp = new Date().toLocaleTimeString();
    const newEntry: ScoreHistory = {
      step: stepCounterRef.current,
      score,
      timestamp,
      improvement
    };
    setScoreHistory(prev => [...prev, newEntry]);
  };

  // Add matrix to history
  const addMatrixToHistory = (matrix: number[][]) => {
    const timestamp = new Date().toLocaleTimeString();
    const newEntry: MatrixHistory = {
      step: stepCounterRef.current,
      matrix: matrix.map(row => [...row]),
      timestamp
    };
    setMatrixHistory(prev => [...prev, newEntry]);
  };

  // Generate plots data for current step
  const generatePlotsData = async (H: number[][], G: number[][], stepNum: number) => {
    try {
      const response = await fetch(buildApiUrl('/api/generate_plots_data'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          H,
          G,
          step_size: parseFloat(stepSize) || 0.01
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }

      const stepPlotsData: StepPlotsData = {
        step: stepNum,
        plots_data: data.plots_data,
        matrix_size: data.matrix_size,
        timestamp: new Date().toLocaleTimeString()
      };
      
      setStepPlotsHistory(prev => [...prev, stepPlotsData]);
    } catch (error) {
      console.error('Error generating plots data:', error);
    }
  };

  // Check if matrices are symmetric
  const isSymmetricMatrix = (matrix: number[][]): boolean => {
    const n = matrix.length;
    for (let i = 0; i < n; i++) {
      if (matrix[i].length !== n) return false;
      for (let j = 0; j < n; j++) {
        if (Math.abs(matrix[i][j] - matrix[j][i]) > 1e-10) {
          return false;
        }
      }
    }
    return true;
  };

  // Calculate Sidorenko score
  const calculateSidorenko = async () => {
    try {
      setIsCalculating(true);
      setError('');

      const { H, G } = getCurrentMatrices();
      
      if (!H || !G) {
        throw new Error('Invalid matrix format. Please use valid JSON array format.');
      }

      // Validate matrices are symmetric
      if (!isSymmetricMatrix(H)) {
        throw new Error('H matrix must be symmetric');
      }
      if (!isSymmetricMatrix(G)) {
        throw new Error('G matrix must be symmetric');
      }

      const response = await fetch(buildApiUrl('/api/calculate'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ H, G }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Calculation failed');
      }

      const data = await response.json();
      setResults(data);
      
      // Add to score history
      if (data.score !== undefined && !isNaN(data.score)) {
        addScoreToHistory(data.score);
      }
      
      // Add initial matrix to history if empty
      if (matrixHistory.length === 0) {
        addMatrixToHistory(G);
      }
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setResults(null);
    } finally {
      setIsCalculating(false);
    }
  };

  // Single optimization step
  const optimizeStep = async () => {
    const matrices = getCurrentMatrices();
    if (!matrices) return;

    const { H, G } = matrices;

    setIsCalculating(true);
    setError('');

    try {
      const response = await fetch(buildApiUrl('/api/optimize_step'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          H,
          G,
          step_size: parseFloat(stepSize) || 0.01
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }

      if (data.success) {
        // Update G matrix
        setGMatrixText(JSON.stringify(data.new_matrix, null, 2));
        
        // Add to histories with current step number
        addScoreToHistory(data.new_score, data.optimization_info.improvement);
        addMatrixToHistory(data.new_matrix);
        
        // Generate plots data for this step
        if (H && data.new_matrix) {
          await generatePlotsData(H, data.new_matrix, stepCounterRef.current);
        }

        // Add optimization step info
        const optimizationStep: OptimizationStep = {
          step: stepCounterRef.current,
          position: data.optimization_info.position,
          direction: data.optimization_info.direction,
          improvement: data.optimization_info.improvement,
          score: data.new_score,
          old_value: data.optimization_info.old_value,
          new_value: data.optimization_info.new_value
        };
        setOptimizationSteps(prev => [...prev, optimizationStep]);

        // Increment step counter for next operation
        stepCounterRef.current += 1;

        setResults({
          score: data.new_score,
          details: data.details
        });

      } else {
        setError(data.message || 'Optimization failed');
      }

    } catch (error) {
      console.error('Optimization error:', error);
      setError(error instanceof Error ? error.message : 'An unknown error occurred');
    } finally {
      setIsCalculating(false);
    }
  };

  // Run multiple optimization steps
  const runMultipleSteps = async () => {
    const matrices = getCurrentMatrices();
    if (!matrices) return;

    setIsOptimizing(true);
    setError('');
    
    const numSteps = parseInt(numberOfSteps) || 10;
    let consecutiveFailures = 0;
    const maxConsecutiveFailures = 3;

    // Track current matrices locally to avoid React state update delays
    let currentH = matrices.H;
    let currentG = matrices.G;

    for (let i = 0; i < numSteps && !optimizationShouldStop.current; i++) {
      try {
        if (!currentH || !currentG) break;

        const response = await fetch(buildApiUrl('/api/optimize_step'), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            H: currentH,
            G: currentG,
            step_size: parseFloat(stepSize) || 0.01
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        
        if (data.error) {
          consecutiveFailures++;
          console.warn(`Step ${i + 1} failed: ${data.error}`);
          
          if (consecutiveFailures >= maxConsecutiveFailures) {
            setError(`Stopped after ${maxConsecutiveFailures} consecutive failures. Last error: ${data.error}`);
            break;
          }
          continue;
        }

        if (data.success) {
          consecutiveFailures = 0; // Reset failure counter
          
          // Update local G matrix for next iteration (immediate, synchronous)
          currentG = data.new_matrix;
          
          // Update React state (asynchronous, for UI display)
          setGMatrixText(JSON.stringify(data.new_matrix, null, 2));
          
          // Add to histories with current step number
          addScoreToHistory(data.new_score, data.optimization_info.improvement);
          addMatrixToHistory(data.new_matrix);
          
          // Generate plots data for this step
          if (currentH && data.new_matrix) {
            await generatePlotsData(currentH, data.new_matrix, stepCounterRef.current);
          }

          // Add optimization step info
          const optimizationStep: OptimizationStep = {
            step: stepCounterRef.current,
            position: data.optimization_info.position,
            direction: data.optimization_info.direction,
            improvement: data.optimization_info.improvement,
            score: data.new_score,
            old_value: data.optimization_info.old_value,
            new_value: data.optimization_info.new_value
          };
          setOptimizationSteps(prev => [...prev, optimizationStep]);

          // Increment step counter for next operation
          stepCounterRef.current += 1;

          setResults({
            score: data.new_score,
            details: data.details
          });

          // Small delay to prevent overwhelming the server
          await new Promise(resolve => setTimeout(resolve, 50));

        } else {
          consecutiveFailures++;
          console.warn(`Step ${i + 1} failed: ${data.message || 'No improvement found'}`);
          
          if (consecutiveFailures >= maxConsecutiveFailures) {
            setError(`Stopped after ${maxConsecutiveFailures} consecutive failures. Last message: ${data.message || 'No improvement found'}`);
            break;
          }
        }

      } catch (error) {
        consecutiveFailures++;
        console.error(`Error in step ${i + 1}:`, error);
        
        if (consecutiveFailures >= maxConsecutiveFailures) {
          setError(`Stopped after ${maxConsecutiveFailures} consecutive failures. Last error: ${error instanceof Error ? error.message : 'Unknown error'}`);
          break;
        }
      }
    }

    setIsOptimizing(false);
  };

  // Stop optimization
  const stopOptimization = () => {
    optimizationShouldStop.current = true;
    setIsOptimizing(false);
  };

  // Initial calculation
  useEffect(() => {
    calculateSidorenko();
  }, []);

  // Render matrix stack visualization
  const renderMatrixStack = () => {
    if (matrixHistory.length === 0) return null;

    return (
      <div className="matrix-stack">
        <h4>Matrix Evolution Stack</h4>
        <div className="matrix-stack-container">
          {matrixHistory.slice(-10).map((entry, index) => { // Show last 10 steps
            const actualIndex = matrixHistory.length - 10 + index;
            if (actualIndex < 0) return null;
            
            // Find the corresponding optimization step for this matrix entry
            const optimizationStep = optimizationSteps.find(step => step.step === entry.step);
            
            return (
              <div key={entry.step} className="matrix-step">
                <div className="step-header">
                  <span className="step-number">Step {entry.step}</span>
                  <span className="step-time">{entry.timestamp}</span>
                </div>
                
                {/* Show what changed */}
                {optimizationStep && optimizationStep.old_value !== undefined && optimizationStep.new_value !== undefined && (
                  <div className="change-note">
                    G[{optimizationStep.position[0]},{optimizationStep.position[1]}] changed from {optimizationStep.old_value.toFixed(3)} to {optimizationStep.new_value.toFixed(3)}
                  </div>
                )}
                
                <div className="matrix-display">
                  {entry.matrix.map((row, i) => (
                    <div key={i} className="matrix-row">
                      {row.map((value, j) => {
                        // Highlight the changed cell
                        const isChangedCell = optimizationStep && 
                          ((optimizationStep.position[0] === i && optimizationStep.position[1] === j) ||
                           (optimizationStep.position[0] === j && optimizationStep.position[1] === i));
                        
                        return (
                          <span 
                            key={j} 
                            className={`matrix-value ${isChangedCell ? 'changed-cell' : ''}`}
                          >
                            {value.toFixed(3)}
                          </span>
                        );
                      })}
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  // Render step plots (plot of plots for each optimization step)
  const renderStepPlots = () => {
    if (stepPlotsHistory.length === 0) {
      return (
        <div className="empty-state">
          <p>No step analysis data available. Run optimization steps to see parameter analysis plots.</p>
        </div>
      );
    }

    return (
      <div className="step-plots-container">
        <h4>Parameter Analysis History</h4>
        <div className="steps-list">
          {stepPlotsHistory.slice().reverse().map((stepPlots) => {
            const plotsData = stepPlots.plots_data;
            const matrixSize = stepPlots.matrix_size;

            return (
              <div key={stepPlots.step} className="step-analysis">
                <div className="step-header">
                  <h5>Step {stepPlots.step} Analysis</h5>
                  <span className="step-time">{stepPlots.timestamp}</span>
                </div>
                <div className="plots-of-plots">
                  <div 
                    className="plots-grid" 
                    style={{ 
                      gridTemplateColumns: `repeat(${matrixSize}, 1fr)` 
                    }}
                  >
                    {Array.from({ length: matrixSize }, (_, i) =>
                      Array.from({ length: matrixSize }, (_, j) => {
                        const key = `${stepPlots.step}-${i},${j}`;
                        const upperTriangleKey = i <= j ? `${i},${j}` : `${j},${i}`; // Get upper triangle equivalent
                        const plotData = plotsData[upperTriangleKey];

                        if (i <= j && plotData) {
                          // Upper triangle or diagonal - show actual plot
                          const chartData = plotData.values.map((value, idx) => ({
                            value,
                            score: plotData.scores[idx]
                          }));

                          return (
                            <div key={key} className="single-plot">
                              <div className="plot-title">{plotData.title}</div>
                              <ResponsiveContainer width="100%" height={150}>
                                <LineChart data={chartData}>
                                  <CartesianGrid strokeDasharray="3 3" />
                                  <XAxis 
                                    dataKey="value" 
                                    type="number"
                                    scale="linear"
                                    domain={['dataMin', 'dataMax']}
                                    tickFormatter={(value) => value.toFixed(3)}
                                  />
                                  <YAxis tickFormatter={(value) => value.toFixed(4)} />
                                  <Tooltip 
                                    formatter={(value: any, name: string) => [
                                      typeof value === 'number' ? value.toFixed(6) : value, 
                                      name === 'score' ? 'Sidorenko Score' : name
                                    ]}
                                    labelFormatter={(value) => `G[${i},${j}] = ${typeof value === 'number' ? value.toFixed(4) : value}`}
                                  />
                                  <Line 
                                    type="monotone" 
                                    dataKey="score" 
                                    stroke="#2563eb" 
                                    strokeWidth={2}
                                    dot={{ r: 3 }}
                                  />
                                </LineChart>
                              </ResponsiveContainer>
                            </div>
                          );
                        } else if (i > j && plotData) {
                          // Lower triangle - duplicate the plot from upper triangle
                          const chartData = plotData.values.map((value, idx) => ({
                            value,
                            score: plotData.scores[idx]
                          }));

                          return (
                            <div key={key} className="single-plot">
                              <div className="plot-title">G[{i},{j}] = G[{j},{i}]</div>
                              <ResponsiveContainer width="100%" height={150}>
                                <LineChart data={chartData}>
                                  <CartesianGrid strokeDasharray="3 3" />
                                  <XAxis 
                                    dataKey="value" 
                                    type="number"
                                    scale="linear"
                                    domain={['dataMin', 'dataMax']}
                                    tickFormatter={(value) => value.toFixed(3)}
                                  />
                                  <YAxis tickFormatter={(value) => value.toFixed(4)} />
                                  <Tooltip 
                                    formatter={(value: any, name: string) => [
                                      typeof value === 'number' ? value.toFixed(6) : value, 
                                      name === 'score' ? 'Sidorenko Score' : name
                                    ]}
                                    labelFormatter={(value) => `G[${i},${j}] = ${typeof value === 'number' ? value.toFixed(4) : value}`}
                                  />
                                  <Line 
                                    type="monotone" 
                                    dataKey="score" 
                                    stroke="#2563eb" 
                                    strokeWidth={2}
                                    dot={{ r: 3 }}
                                  />
                                </LineChart>
                              </ResponsiveContainer>
                            </div>
                          );
                        } else {
                          // No data available
                          return (
                            <div key={key} className="single-plot">
                              <div className="plot-title">G[{i},{j}]</div>
                              <div className="empty-state">
                                <p>No data</p>
                              </div>
                            </div>
                          );
                        }
                      })
                    ).flat()}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Sidorenko's Conjecture Visualization Tool</h1>
        <div className="header-controls">
          <button onClick={loadExampleData} className="control-button">
            Load Example
          </button>
          <button onClick={calculateSidorenko} disabled={isCalculating} className="control-button primary">
            {isCalculating ? 'Calculating...' : 'Calculate'}
          </button>
          <button onClick={() => setShowPlots(!showPlots)} className="control-button">
            {showPlots ? 'Hide Plots' : 'Show Plots'}
          </button>
          <button onClick={clearAllHistory} className="control-button">
            Clear History
          </button>
        </div>
      </header>

      <div className="main-content">
        {/* Configuration Panel */}
        <div className="input-panel">
          <h2>Configuration</h2>
          
          {/* Matrix Inputs - Back to text format */}
          <div className="matrices-container">
            <div className="matrix-section">
              <h3>Pattern Graph H (Adjacency Matrix)</h3>
              <textarea
                value={HMatrixText}
                onChange={(e) => setHMatrixText(e.target.value)}
                className="matrix-input"
                placeholder="Enter 2D array like [[0,1,0],[1,0,1],[0,1,0]]"
                rows={4}
              />
              <p className="matrix-help">Enter H as a JSON 2D array with 0s and 1s</p>
            </div>

            <div className="matrix-section">
              <h3>Host Graph G (Weighted Adjacency Matrix)</h3>
              <textarea
                value={GMatrixText}
                onChange={(e) => setGMatrixText(e.target.value)}
                className="matrix-input"
                placeholder="Enter 2D array like [[0.5,0.3,0.2],[0.3,0.4,0.1],[0.2,0.1,0.6]]"
                rows={4}
              />
              <p className="matrix-help">Enter G as a JSON 2D array with decimal values</p>
            </div>
          </div>

          {/* Optimization Controls */}
          <div className="optimization-section">
            <h3>Optimization Controls</h3>
            <div className="input-group">
              <label>Step Size (Epsilon):</label>
              <input
                type="text"
                value={stepSize}
                onChange={(e) => setStepSize(e.target.value)}
                className="step-size-input"
                placeholder="0.01"
              />
            </div>
            
            <div className="input-group">
              <label>Number of Steps:</label>
              <input
                type="text"
                value={numberOfSteps}
                onChange={(e) => setNumberOfSteps(e.target.value)}
                className="step-size-input"
                placeholder="10"
              />
            </div>
            
            <div className="button-group">
              <button onClick={optimizeStep} disabled={isOptimizing} className="control-button">
                Single Step
              </button>
              <button onClick={runMultipleSteps} disabled={isOptimizing} className="control-button">
                Run {numberOfSteps} Steps
              </button>
              <button onClick={stopOptimization} disabled={!isOptimizing} className="control-button danger">
                Stop
              </button>
            </div>
            {isOptimizing && <p className="status-text">Running optimization...</p>}
          </div>
        </div>

        {/* Results Panel */}
        <div className="visualization-panel">
          <h2>Results & Visualization</h2>
          
          {error && (
            <div className="error-message">
              <h3>Error</h3>
              <p>{error}</p>
            </div>
          )}

          {results && !error && (
            <div className="results-display">
              <div className="score-display">
                <h3>Sidorenko Score</h3>
                <div className={`score-value ${results.score > 0 ? 'positive' : results.score < 0 ? 'negative' : ''}`}>
                  {results.score?.toFixed(6) || 'N/A'}
                </div>
                {results.score < 0 && (
                  <div className="counterexample-label">
                    ⚠ Potential Counterexample Found!
                  </div>
                )}
              </div>

              {results.details && (
                <div className="computation-details">
                  <h4>Computation Details</h4>
                  <div className="details-grid">
                    <div className="detail-item">
                      <span className="label">Homomorphism Count:</span>
                      <span className="value">{results.details.hom_count?.toFixed(6) || 'N/A'}</span>
                    </div>
                    <div className="detail-item">
                      <span className="label">Homomorphism Density (t_H_G):</span>
                      <span className="value">{results.details.t_H_G?.toFixed(6) || 'N/A'}</span>
                    </div>
                    <div className="detail-item">
                      <span className="label">Edge Density (p):</span>
                      <span className="value">{results.details.edge_density_p?.toFixed(6) || 'N/A'}</span>
                    </div>
                    <div className="detail-item">
                      <span className="label">p^|E(H)|:</span>
                      <span className="value">{results.details.p_power_edges?.toFixed(6) || 'N/A'}</span>
                    </div>
                  </div>
                </div>
              )}

              {showPlots && (
                <div className="plots-layout">
                  {/* Left side: Matrix Stack */}
                  <div className="left-plots">
                    {renderMatrixStack()}
                    
                    {/* Score Evolution */}
                    {scoreHistory.length > 0 && (
                      <div className="plot-section">
                        <h4>Score Evolution</h4>
                        <ResponsiveContainer width="100%" height={200}>
                          <LineChart data={scoreHistory}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="step" />
                            <YAxis />
                            <Tooltip 
                              formatter={(value: any, name: string) => [value.toFixed(6), name]}
                              labelFormatter={(step) => `Step: ${step}`}
                            />
                            <Line 
                              type="monotone" 
                              dataKey="score" 
                              stroke="#667eea" 
                              strokeWidth={2}
                              dot={{ fill: '#667eea', strokeWidth: 2, r: 4 }}
                              name="Sidorenko Score"
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    )}
                  </div>

                  {/* Right side: Step Plots Analysis */}
                  <div className="right-plots">
                    {renderStepPlots()}
                  </div>
                </div>
              )}
            </div>
          )}

          {!results && !error && !isCalculating && (
            <div className="empty-state">
              <p>Click "Calculate" to compute the Sidorenko score and view visualizations</p>
            </div>
          )}

          {isCalculating && (
            <div className="loading-state">
              <div className="loading-spinner"></div>
              <p>Calculating Sidorenko score...</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
