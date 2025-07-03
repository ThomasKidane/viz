# Sidorenko Conjecture Visualization Tool

A powerful web-based tool for exploring Sidorenko's conjecture through interactive visualization, optimization, and analysis of graph homomorphism densities.

## Overview

This tool allows researchers to:
- Compute Sidorenko scores for arbitrary graph patterns and host graphs
- Visualize how parameter changes affect homomorphism densities
- Run optimization algorithms to search for potential counterexamples to Sidorenko's conjecture
- Track the evolution of matrices through optimization steps
- Analyze parameter sensitivity with interactive plots

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
- High-precision homomorphism counting using custom graph libraries
- Sidorenko score computation with detailed mathematical breakdown
- Support for weighted and unweighted graphs

### 🎯 Optimization Engine
- **Single-step optimization**: Find the best parameter change for one iteration
- **Multi-step optimization**: Run continuous optimization with stopping conditions
- **Global search**: Test all valid parameter values within specified ranges
- **Matrix bounds enforcement**: All values automatically clamped to [0,1] range

### 📊 Interactive Visualization
- **Real-time score display** with counterexample detection
- **Matrix evolution tracking** with change highlighting
- **Parameter analysis plots** showing how each matrix entry affects the score
- **Score evolution charts** throughout the optimization process
- **Scrollable history** of all optimization steps

### 🔍 Analysis Tools
- **Parameter sensitivity analysis**: See how changing each matrix entry affects the Sidorenko score
- **Change tracking**: Visual highlighting of modified matrix entries
- **Historical comparison**: Review all previous optimization steps
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

3. Install the custom homomorphism counting library:
```bash
cd CountHomLib
pip install .
cd ..
```

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

### 1. Input Matrices

**Pattern Graph H**: Enter the adjacency matrix of your pattern graph (should be unweighted, 0s and 1s)
```json
[[0, 1], [1, 0]]
```

**Host Graph G**: Enter the adjacency matrix of your host graph (values between 0 and 1)
```json
[[0.5, 0.8], [0.8, 0.3]]
```

### 2. Compute Sidorenko Score

Click **"Calculate Sidorenko Score"** to see:
- Current Sidorenko score
- Homomorphism count details
- Edge density calculations
- Counterexample detection

### 3. Run Optimization

**Single Step Optimization**:
- Click **"Optimize Single Step"** to find the best parameter change
- View the updated matrix and improved score

**Multi-Step Optimization**:
1. Set step size (e.g., 0.01) and number of steps (e.g., 100)
2. Click **"Run Multiple Steps"**
3. Watch real-time optimization progress
4. Use **"Stop Optimization"** to halt early

### 4. Analyze Results

- **Matrix Evolution**: Review all matrices from the optimization history
- **Parameter Analysis**: Examine plots showing how each parameter affects the score
- **Score Charts**: Track score improvements over time
- **Change Detection**: See exactly which matrix entries were modified

## Example Workflow

1. **Start with a simple case**:
   - H: `[[0,1],[1,0]]` (edge graph)
   - G: `[[0.5,0.8],[0.8,0.3]]` (random values)

2. **Calculate initial score**: Often positive for random matrices

3. **Run optimization**: Use step size 0.01 for 50 steps

4. **Analyze results**: Check if score becomes negative (potential counterexample)

## Project Structure

```
viz/
├── app.py                      # Flask backend with optimization algorithms
├── requirements.txt            # Python dependencies
├── CountHomLib/               # High-performance homomorphism counting
│   ├── src/                   # C++ source code with Python bindings
│   └── setup.py               # Library installation script
├── customgraphhomo/           # Custom homomorphism computation
│   └── counthomo.py           # Python implementation for verification
├── frontend/                  # React TypeScript frontend
│   ├── src/
│   │   ├── App.tsx            # Main application component
│   │   ├── App.css            # Styling and layout
│   │   └── config.ts          # API configuration
│   ├── package.json           # Node.js dependencies
│   └── public/                # Static assets
└── README.md                  # This file
```

## Technical Details

### Backend Architecture
- **Flask** REST API with CORS support
- **SidorenkoCalculator** class for core mathematical computations
- **Optimization algorithms** using gradient-free parameter exploration
- **High-precision arithmetic** using Python's `Decimal` type
- **Matrix validation** ensuring symmetry and [0,1] bounds

### Frontend Architecture
- **React TypeScript** for type-safe UI development
- **Recharts** for interactive data visualization
- **Real-time updates** during optimization
- **Responsive design** for various screen sizes
- **Local state management** for optimization tracking

### Mathematical Implementation
- Homomorphism counting via dynamic programming on tree decompositions
- Efficient algorithms with complexity O(|V(G)|^{tw(H)+1})
- Special optimizations for tree patterns
- Support for both exact and weighted graph homomorphisms

## API Endpoints

- `POST /api/calculate` - Compute Sidorenko score for given H and G
- `POST /api/optimize_step` - Find optimal single parameter change
- `POST /api/generate_plots_data` - Generate parameter analysis plots

## Contributing

This tool is designed for mathematical research into Sidorenko's conjecture. Contributions are welcome for:
- Additional optimization algorithms
- New visualization features
- Performance improvements
- Extended mathematical functionality

## Research Applications

This tool has been used to:
- Explore potential counterexamples to Sidorenko's conjecture
- Analyze parameter sensitivity in extremal graph theory
- Visualize optimization landscapes for graph homomorphism problems
- Generate test cases for theoretical analysis

## License

This project is provided for research and educational purposes. Please cite appropriately if used in academic work.

## Acknowledgments

- Built on efficient graph homomorphism counting algorithms
- Uses tree decomposition methods for optimal complexity
- Incorporates high-precision arithmetic for reliable computations
- Frontend design focused on mathematical research workflows 