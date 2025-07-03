# Sidorenko Conjecture Visualization Tool

A powerful web-based tool for exploring Sidorenko's conjecture through interactive visualization, optimization, and analysis of graph homomorphism densities.

## Overview

This tool allows researchers to:
- Compute Sidorenko scores for arbitrary graph patterns and host graphs
- Visualize how parameter changes affect homomorphism densities
- Run optimization algorithms to search for potential counterexamples to Sidorenko's conjecture
- Track the evolution of matrices through optimization steps
- Analyze parameter sensitivity with interactive plots
- **Switch between high-performance CountHomLib and custom implementations**

## Mathematical Background

**Sidorenko's Conjecture** (1993) states that for any bipartite graph H and any graph G:

```
t_H(G) ≥ (t_K₂(G))^|E(H)|
```

Where:
- `t_H(G)` is the homomorphism density of H in G
- `t_K₂(G)` is the edge density of G  
- `|E(H)|` is the number of edges in H

The **Sidorenko score** is defined as: `t_H(G) - (t_K₂(G))^|E(H)|`

- A **negative score** indicates a potential counterexample to the conjecture
- A **positive score** supports the conjecture for that specific case

## Features

### 🧮 Core Calculations
- **Dual Implementation Support**: Choose between CountHomLib (high-performance C++) and custom Python implementation
- High-precision homomorphism counting using state-of-the-art algorithms
- Sidorenko score computation with detailed mathematical breakdown
- Support for weighted graphons and unweighted graphs

### ⚡ CountHomLib Integration
- **High-Performance Computing**: Utilize efficient C++ implementation with Python bindings
- **Tree Decomposition Algorithms**: Optimal complexity O(|V(G)|^{tw(H)+1}) using nice tree decompositions
- **Graphon Support**: Direct homomorphism density calculation for weighted graphs
- **Automatic Fallback**: Seamlessly switches to custom implementation if CountHomLib is unavailable

### 🎯 Optimization Engine
- **Single-step optimization**: Find the best parameter change for one iteration
- **Multi-step optimization**: Run continuous optimization with stopping conditions
- **Global search**: Test all valid parameter values within specified ranges
- **Matrix bounds enforcement**: All values automatically clamped to [0,1] range
- **Implementation switching**: Use either CountHomLib or custom implementation for optimization

### 📊 Interactive Visualization
- **Real-time score display** with counterexample detection
- **Matrix evolution tracking** with change highlighting
- **Parameter analysis plots** showing how each matrix entry affects the score
- **Score evolution charts** throughout the optimization process
- **Scrollable history** of all optimization steps
- **Implementation indicator**: See which computation method was used

### 🔍 Analysis Tools
- **Parameter sensitivity analysis**: See how changing each matrix entry affects the Sidorenko score
- **Change tracking**: Visual highlighting of modified matrix entries
- **Historical comparison**: Review all previous optimization steps
- **Performance comparison**: Compare results between different implementations
- **Export capabilities**: Save matrices and results for further analysis

## Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd viz
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install CountHomLib (Optional but Recommended):
```bash
cd CountHomLib
pip install .
cd ..
```

**Note**: If CountHomLib installation fails, the tool will automatically use the custom implementation.

4. Start the Flask backend:
```bash
python app.py
```
The backend will run on `http://localhost:5002`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```
The frontend will open at `http://localhost:3000`

## Usage

### 1. Choose Implementation

Click the **"Using CountHomLib"** or **"Using Custom"** button in the header to switch between:
- **CountHomLib**: High-performance C++ implementation (🚀 recommended for large graphs)
- **Custom**: Python implementation (🐌 works without additional dependencies)

### 2. Input Matrices

**Pattern Graph H**: Enter the adjacency matrix of your pattern graph (should be unweighted, 0s and 1s)
```json
[[0, 1], [1, 0]]
```

**Host Graph G**: Enter the adjacency matrix of your host graph (values between 0 and 1)
```json
[[0.5, 0.8], [0.8, 0.3]]
```

### 3. Compute Sidorenko Score

Click **"Calculate Sidorenko Score"** to see:
- Current Sidorenko score
- Homomorphism count details
- Edge density calculations
- Counterexample detection
- **Implementation used** (CountHomLib or Custom)

### 4. Run Optimization

**Single Step Optimization**:
- Click **"Optimize Single Step"** to find the best parameter change
- View the updated matrix and improved score

**Multi-Step Optimization**:
1. Set step size (e.g., 0.01) and number of steps (e.g., 100)
2. Click **"Run Multiple Steps"**
3. Watch real-time optimization progress
4. Use **"Stop Optimization"** to halt early

### 5. Analyze Results

- **Matrix Evolution**: Review all matrices from the optimization history
- **Parameter Analysis**: Examine plots showing how each parameter affects the score
- **Score Charts**: Track score improvements over time
- **Change Detection**: See exactly which matrix entries were modified
- **Implementation Comparison**: Switch between implementations to compare results

## Performance Comparison

| Feature | CountHomLib | Custom Implementation |
|---------|-------------|----------------------|
| **Speed** | Very Fast (C++) | Moderate (Python) |
| **Memory** | Efficient | Higher usage |
| **Complexity** | O(\|V(G)\|^{tw(H)+1}) | O(\|V(G)\|^{\|V(H)\|}) |
| **Dependencies** | Requires installation | Built-in |
| **Precision** | High (double) | Very High (Decimal) |
| **Graph Size** | Large graphs supported | Best for small graphs |

**Recommendation**: Use CountHomLib for large graphs and intensive computations. Use Custom implementation for verification or when CountHomLib is unavailable.

## Example Workflow

1. **Start with CountHomLib**: Click the toggle to ensure high performance
2. **Load example data**: H: `[[0,1,0],[1,0,1],[0,1,0]]` (path graph), G: `[[0.5,0.3,0.2],[0.3,0.4,0.1],[0.2,0.1,0.6]]`
3. **Calculate initial score**: Often positive for random matrices
4. **Run optimization**: Use step size 0.01 for 50 steps
5. **Compare implementations**: Toggle to Custom to verify results
6. **Analyze results**: Check if score becomes negative (potential counterexample)

## Project Structure

```
viz/
├── app.py                      # Flask backend with dual implementation support
├── requirements.txt            # Python dependencies
├── counthomo.py               # Custom Python implementation (fallback)
├── CountHomLib/               # High-performance C++ library
│   ├── src/                   # C++ source with tree decomposition algorithms
│   ├── setup.py               # Library installation script
│   └── README.md              # CountHomLib documentation
├── frontend/                  # React TypeScript frontend
│   ├── src/
│   │   ├── App.tsx            # Main app with implementation toggle
│   │   ├── App.css            # Styling and responsive design
│   │   └── config.ts          # API configuration
│   ├── package.json           # Node.js dependencies
│   └── public/                # Static assets
└── README.md                  # This file
```

## Technical Details

### Backend Architecture
- **Flask** REST API with CORS support
- **Dual Implementation**: Automatic detection and switching between CountHomLib and custom code
- **SidorenkoCalculator** class supporting both calculation methods
- **Optimization algorithms** using gradient-free parameter exploration
- **Matrix validation** ensuring symmetry and [0,1] bounds
- **Error handling** with graceful fallback between implementations

### Frontend Architecture
- **React TypeScript** for type-safe UI development
- **Implementation Toggle**: Real-time switching between calculation methods
- **Recharts** for interactive data visualization
- **Real-time updates** during optimization
- **Status indicators** showing which implementation is active
- **Responsive design** for various screen sizes

### Mathematical Implementation

#### CountHomLib (Recommended)
- Tree decomposition-based dynamic programming
- Complexity: O(|V(G)|^{tw(H)+1}) where tw(H) is treewidth of H
- Optimized for bipartite graphs and sparse patterns
- Direct graphon homomorphism density calculation
- Handles large graphs efficiently

#### Custom Implementation (Fallback)
- Brute force enumeration for graphon homomorphisms
- Complexity: O(|V(G)|^{|V(H)|})
- High-precision arithmetic using Python's Decimal
- Best for small graphs and verification
- Full Python implementation with no external dependencies

## API Endpoints

- `GET /api/counthomlib_status` - Check CountHomLib availability
- `POST /api/calculate` - Compute Sidorenko score (with `use_countHomLib` parameter)
- `POST /api/optimize_step` - Find optimal single parameter change
- `POST /api/generate_plots_data` - Generate parameter analysis plots

All endpoints support the `use_countHomLib` boolean parameter to choose implementation.

## Contributing

This tool is designed for mathematical research into Sidorenko's conjecture. Contributions are welcome for:
- Additional optimization algorithms
- New visualization features
- Performance improvements
- Extended mathematical functionality
- CountHomLib integration enhancements

## Research Applications

This tool has been used to:
- Explore potential counterexamples to Sidorenko's conjecture
- Analyze parameter sensitivity in extremal graph theory
- Visualize optimization landscapes for graph homomorphism problems
- Generate test cases for theoretical analysis
- Compare algorithmic approaches to homomorphism counting

## Dependencies

### Required (Backend)
- Flask, Flask-CORS, numpy

### Optional (High Performance)
- CountHomLib: High-performance C++ library with Python bindings
- pybind11: Required for CountHomLib compilation

### Frontend
- React, TypeScript, Recharts
- Responsive CSS for multi-device support

## License

This project is provided for research and educational purposes. Please cite appropriately if used in academic work.

## Acknowledgments

- Built on efficient graph homomorphism counting algorithms from CountHomLib
- Uses tree decomposition methods for optimal complexity
- Incorporates both high-performance C++ and high-precision Python implementations
- Frontend design focused on mathematical research workflows
- Supports both exact computation and approximation methods 