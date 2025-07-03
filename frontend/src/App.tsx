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

interface CalculationDetails {
  hom_count: number;
  t_H_G: number;
  edge_density_p: number;
  p_power_edges: number;
  num_edges_H: number;
  sidorenko_score: number;
  implementation?: string;
  error?: string;
}

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
  const [results, setResults] = useState<{ score: number; details: CalculationDetails } | null>(null);
  const [error, setError] = useState<string>('');
  const [isCalculating, setIsCalculating] = useState(false);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [optimizationProgress, setOptimizationProgress] = useState<{
    currentStep: number;
    totalSteps: number;
    consecutiveNoImprovements: number;
    lastImprovement?: number;
    status: 'running' | 'error';
  } | null>(null);
  
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

  // CountHomLib related state
  const [useCountHomLib, setUseCountHomLib] = useState(false);
  const [countHomLibAvailable, setCountHomLibAvailable] = useState(false);
  const [countHomLibStatus, setCountHomLibStatus] = useState<string>('Checking...');

  // Check CountHomLib availability on startup
  useEffect(() => {
    const checkCountHomLibStatus = async () => {
      try {
        const response = await fetch(buildApiUrl('/api/counthomlib_status'));
        const data = await response.json();
        setCountHomLibAvailable(data.available);
        setCountHomLibStatus(data.message);
      } catch (error) {
        setCountHomLibAvailable(false);
        setCountHomLibStatus('Error checking CountHomLib status');
      }
    };
    
    checkCountHomLibStatus();
  }, []);

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
          step_size: parseFloat(stepSize) || 0.01,
          use_countHomLib: useCountHomLib
        }),
      });

      const data = await response.json();

      // Handle server errors (400, 500, etc.) that return JSON with error messages
      if (!response.ok) {
        const errorMessage = data.error || `HTTP ${response.status}: ${response.statusText}`;
        throw new Error(errorMessage);
      }
      
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
        body: JSON.stringify({ H, G, use_countHomLib: useCountHomLib }),
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
          step_size: parseFloat(stepSize) || 0.01,
          use_countHomLib: useCountHomLib
        }),
      });

      const data = await response.json();

      // Handle server errors (400, 500, etc.) that return JSON with error messages
      if (!response.ok) {
        const errorMessage = data.error || `HTTP ${response.status}: ${response.statusText}`;
        throw new Error(errorMessage);
      }
      
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
    let consecutiveNoImprovements = 0;
    const maxConsecutiveFailures = 10; // Only stop on actual errors
    // Removed maxConsecutiveNoImprovements - never stop for "no improvement"

    // Initialize progress tracking
    setOptimizationProgress({
      currentStep: 0,
      totalSteps: numSteps,
      consecutiveNoImprovements: 0,
      status: 'running'
    });

    // Track current matrices locally to avoid React state update delays
    let currentH = matrices.H;
    let currentG = matrices.G;

    for (let i = 0; i < numSteps && !optimizationShouldStop.current; i++) {
      // Update progress
                setOptimizationProgress(prev => prev ? {
            ...prev,
            currentStep: i + 1,
            consecutiveNoImprovements,
            status: 'running' // Always keep running, never mark as converging
          } : null);

      try {
        if (!currentH || !currentG) break;

        // Debug: Validate matrices before sending
        if (!isSymmetricMatrix(currentH)) {
          console.error('H matrix is not symmetric at step', i + 1, currentH);
          setError(`H matrix became non-symmetric at step ${i + 1}`);
          break;
        }
        if (!isSymmetricMatrix(currentG)) {
          console.error('G matrix is not symmetric at step', i + 1, currentG);
          setError(`G matrix became non-symmetric at step ${i + 1}`);
          break;
        }

        // Debug: Check for invalid values in G matrix and fix small floating point errors
        let hasInvalidValues = false;
        currentG = currentG.map(row => 
          row.map(val => {
            if (isNaN(val)) {
              hasInvalidValues = true;
              return val;
            }
            // Fix small floating point precision errors
            if (val < 0 && val > -1e-10) return 0;
            if (val > 1 && val < 1 + 1e-10) return 1;
            if (val < 0 || val > 1) {
              hasInvalidValues = true;
            }
            return val;
          })
        );
        
        if (hasInvalidValues) {
          console.error('G matrix has invalid values at step', i + 1, currentG);
          setError(`G matrix has values outside [0,1] range or NaN at step ${i + 1}`);
          break;
        }

        const response = await fetch(buildApiUrl('/api/optimize_step'), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            H: currentH,
            G: currentG,
            step_size: parseFloat(stepSize) || 0.01,
            use_countHomLib: useCountHomLib
          }),
        });

        const data = await response.json();

        // Handle server errors (400, 500, etc.) that return JSON with error messages
        if (!response.ok) {
          const errorMessage = data.error || `HTTP ${response.status}: ${response.statusText}`;
          console.error(`API Error at step ${i + 1}:`, {
            status: response.status,
            statusText: response.statusText,
            errorMessage,
            currentH,
            currentG,
            stepSize: parseFloat(stepSize) || 0.01,
            useCountHomLib
          });
          throw new Error(errorMessage);
        }
        
        if (data.error) {
          consecutiveFailures++;
          console.warn(`Step ${i + 1} failed with error: ${data.error}`);
          
          setOptimizationProgress(prev => prev ? { ...prev, status: 'error' } : null);
          
          if (consecutiveFailures >= maxConsecutiveFailures) {
            setError(`Stopped after ${maxConsecutiveFailures} consecutive errors. Last error: ${data.error}`);
            break;
          }
          continue;
        }

        if (data.success) {
          // Reset both counters on success
          consecutiveFailures = 0;
          consecutiveNoImprovements = 0;
          
          // Update progress with improvement
          setOptimizationProgress(prev => prev ? {
            ...prev,
            consecutiveNoImprovements: 0,
            lastImprovement: data.optimization_info.improvement,
            status: 'running'
          } : null);
          
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
          await new Promise(resolve => setTimeout(resolve, 100)); // Increased from 50ms

        } else {
          // Handle "no improvement found" case (not an error, continue searching)
          consecutiveNoImprovements++;
          console.log(`Step ${i + 1}: ${data.message || 'No improvement found'} (${consecutiveNoImprovements} consecutive steps) - continuing search...`);
          
          // Update progress showing we're still searching
          setOptimizationProgress(prev => prev ? {
            ...prev,
            currentStep: i + 1,
            consecutiveNoImprovements,
            status: 'running' // Always keep status as running
          } : null);
          
          // Never stop for "no improvement" - keep exploring the parameter space
          // Add a short delay to prevent overwhelming the server
          await new Promise(resolve => setTimeout(resolve, 150));
        }

      } catch (error) {
        consecutiveFailures++;
        console.error(`Error in step ${i + 1}:`, error);
        
        setOptimizationProgress(prev => prev ? { ...prev, status: 'error' } : null);
        
        if (consecutiveFailures >= maxConsecutiveFailures) {
          setError(`Stopped after ${maxConsecutiveFailures} consecutive errors. Last error: ${error instanceof Error ? error.message : 'Unknown error'}`);
          break;
        }
        
        // Longer delay after errors
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    }

    setIsOptimizing(false);
    setOptimizationProgress(null);
  };

  // Stop optimization
  const stopOptimization = () => {
    optimizationShouldStop.current = true;
    setIsOptimizing(false);
    
    // Provide feedback about stopped optimization
    if (optimizationProgress) {
      const { currentStep, totalSteps, consecutiveNoImprovements } = optimizationProgress;
      let message = `Optimization stopped by user after ${currentStep} steps`;
      
      if (consecutiveNoImprovements > 0) {
        message += `. No improvements found for last ${consecutiveNoImprovements} steps`;
      }
      
      if (currentStep < totalSteps) {
        message += ` (${totalSteps - currentStep} steps remaining)`;
      }
      
      setError(message);
    }
    
    setOptimizationProgress(null);
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
                    G[{optimizationStep.position[0]},{optimizationStep.position[1]}] changed from {optimizationStep.old_value.toFixed(8)} to {optimizationStep.new_value.toFixed(8)}
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
                            {value.toFixed(6)}
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
                                    tickFormatter={(value) => value.toFixed(6)}
                                  />
                                  <YAxis tickFormatter={(value) => value.toFixed(8)} />
                                  <Tooltip 
                                    formatter={(value: any, name: string) => [
                                      typeof value === 'number' ? value.toFixed(12) : value, 
                                      name === 'score' ? 'Sidorenko Score' : name
                                    ]}
                                    labelFormatter={(value) => `G[${i},${j}] = ${typeof value === 'number' ? value.toFixed(8) : value}`}
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
                                    tickFormatter={(value) => value.toFixed(6)}
                                  />
                                  <YAxis tickFormatter={(value) => value.toFixed(8)} />
                                  <Tooltip 
                                    formatter={(value: any, name: string) => [
                                      typeof value === 'number' ? value.toFixed(12) : value, 
                                      name === 'score' ? 'Sidorenko Score' : name
                                    ]}
                                    labelFormatter={(value) => `G[${i},${j}] = ${typeof value === 'number' ? value.toFixed(8) : value}`}
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

  // Toggle CountHomLib usage
  const toggleCountHomLib = () => {
    if (countHomLibAvailable) {
      setUseCountHomLib(!useCountHomLib);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Sidorenko's Conjecture Visualization Tool</h1>
        <div className="header-controls">
          <button onClick={toggleCountHomLib} className={`control-button ${useCountHomLib ? 'primary' : ''}`} disabled={!countHomLibAvailable} title={countHomLibAvailable ? (useCountHomLib ? 'Switch to Custom Implementation' : 'Switch to CountHomLib (High Performance)') : countHomLibStatus}>
            {useCountHomLib ? 'Using CountHomLib' : 'Using Custom'}
          </button>
          <span className="status-text">
            {countHomLibAvailable ? 
              (useCountHomLib ? '🚀 High Performance' : '🐌 Custom Implementation') : 
              '⚠️ CountHomLib Not Available'
            }
          </span>
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
            
            <div className="optimization-help">
              <p><strong>Optimization Notes:</strong></p>
              <ul>
                <li>🏃 <strong>Running:</strong> Continuously searching for better parameters</li>
                <li>🔍 <strong>Exploring:</strong> Keeps searching even when no immediate improvements found</li>
                <li>⚠️ <strong>Issues:</strong> Only stops for actual calculation errors</li>
                <li>The algorithm will run for the full number of steps you specify - never stops for "no improvement"</li>
                <li>Press "Stop Optimization" to halt manually at any time</li>
              </ul>
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
            
            {/* Enhanced optimization status display */}
            {isOptimizing && optimizationProgress && (
              <div className="optimization-status">
                <div className="status-header">
                  <span className="status-text">
                    {optimizationProgress.status === 'running' && '🏃 Optimizing...'}
                    {optimizationProgress.status === 'error' && '⚠️ Issues detected'}
                  </span>
                  <span className="progress-text">
                    Step {optimizationProgress.currentStep} / {optimizationProgress.totalSteps}
                  </span>
                </div>
                
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{ width: `${(optimizationProgress.currentStep / optimizationProgress.totalSteps) * 100}%` }}
                  />
                </div>
                
                <div className="status-details">
                  {optimizationProgress.consecutiveNoImprovements > 0 && (
                    <span className="convergence-info">
                      🔍 Exploring: {optimizationProgress.consecutiveNoImprovements} steps without improvement (continuing search...)
                    </span>
                  )}
                  {optimizationProgress.lastImprovement && (
                    <span className="last-improvement">
                      Last improvement: {optimizationProgress.lastImprovement.toFixed(12)}
                    </span>
                  )}
                </div>
              </div>
            )}
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
                  {results.score?.toFixed(12) || 'N/A'}
                </div>
                {results.score < 0 && (
                  <div className="counterexample-label">
                    ⚠ Potential Counterexample Found!
                  </div>
                )}
              </div>

              {results.details && (
                <div className="computation-details">
                  <h4>Computation Details {results.details.implementation && `(${results.details.implementation})`}</h4>
                  <div className="details-grid">
                    <div className="detail-item">
                      <span className="label">Homomorphism Count:</span>
                      <span className="value">{results.details.hom_count?.toFixed(12) || 'N/A'}</span>
                    </div>
                    <div className="detail-item">
                      <span className="label">Homomorphism Density (t_H_G):</span>
                      <span className="value">{results.details.t_H_G?.toFixed(12) || 'N/A'}</span>
                    </div>
                    <div className="detail-item">
                      <span className="label">Edge Density (p):</span>
                      <span className="value">{results.details.edge_density_p?.toFixed(12) || 'N/A'}</span>
                    </div>
                    <div className="detail-item">
                      <span className="label">p^|E(H)|:</span>
                      <span className="value">{results.details.p_power_edges?.toFixed(12) || 'N/A'}</span>
                    </div>
                    {results.details.implementation && (
                      <div className="detail-item">
                        <span className="label">Implementation:</span>
                        <span className="value">{results.details.implementation}</span>
                      </div>
                    )}
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
                              formatter={(value: any, name: string) => [value.toFixed(12), name]}
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
