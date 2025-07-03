from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from decimal import Decimal, getcontext
import traceback
import logging

# Import the simpler customgraphhomo package
from customgraphhomo.counthomo import WeightedGraph, count_homomorphisms

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def is_symmetric_matrix(matrix):
    """Check if a matrix is symmetric"""
    n = len(matrix)
    for i in range(n):
        if len(matrix[i]) != n:
            return False
        for j in range(n):
            if abs(matrix[i][j] - matrix[j][i]) > 1e-10:
                return False
    return True

def is_valid_probability_matrix(matrix):
    """Check if a matrix has all values in [0, 1] range"""
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            value = matrix[i][j]
            if value < 0 or value > 1:
                return False
    return True

class SidorenkoCalculator:
    def __init__(self, H_matrix, G_matrix):
        """
        Initialize with H (pattern graph) and G (host graph) adjacency matrices.
        
        Args:
            H_matrix: Adjacency matrix for pattern graph H (should be unweighted, all edges = 1)
            G_matrix: Adjacency matrix for host graph G (can be weighted)
        """
        if not is_symmetric_matrix(H_matrix):
            raise ValueError("H matrix must be symmetric")
        if not is_symmetric_matrix(G_matrix):
            raise ValueError("G matrix must be symmetric")
        if not is_valid_probability_matrix(G_matrix):
            raise ValueError("G matrix must have all values between 0 and 1 (inclusive)")
            
        self.H_matrix = H_matrix
        self.G_matrix = G_matrix
        self.n_H = len(H_matrix)
        self.n_G = len(G_matrix)
        
        # Create WeightedGraph objects using the correct API
        # For H: unweighted graph (all edge weights = 1, node weights = 1)
        h_edge_weights = {}
        for i in range(self.n_H):
            h_edge_weights[i] = {}
            for j in range(self.n_H):
                if H_matrix[i][j] != 0:
                    h_edge_weights[i][j] = 1.0  # All edges have weight 1
                else:
                    h_edge_weights[i][j] = 0.0
        
        h_node_weights = {i: 1.0 for i in range(self.n_H)}  # All nodes have weight 1
        self.H_graph = WeightedGraph(edge_weights=h_edge_weights, node_weights=h_node_weights)
        
        # For G: weighted graph from user input
        g_edge_weights = {}
        for i in range(self.n_G):
            g_edge_weights[i] = {}
            for j in range(self.n_G):
                g_edge_weights[i][j] = float(G_matrix[i][j])
        
        g_node_weights = {i: 1.0 for i in range(self.n_G)}  # All nodes have weight 1
        self.G_graph = WeightedGraph(edge_weights=g_edge_weights, node_weights=g_node_weights)
        
        # Count edges in H (for Sidorenko calculation)
        self.num_edges_H = sum(1 for i in range(self.n_H) for j in range(i+1, self.n_H) if H_matrix[i][j] != 0)
    
    def calculate_sidorenko_score(self):
        """
        Calculate the Sidorenko score t_H_G - p^|E(H)|
        where:
        - t_H_G = hom_count / (n_G^n_H) (homomorphism density)
        - p = average of all entries in G matrix (edge density)
        """
        try:
            # Set high precision for calculations
            getcontext().prec = 50
            
            # Count homomorphisms from H to G
            hom_count = count_homomorphisms(self.H_graph, self.G_graph, precision=50)
            
            # Calculate homomorphism density: t_H_G = hom_count / (n_G^n_H)
            n_G_decimal = Decimal(self.n_G)
            n_H_decimal = Decimal(self.n_H)
            
            if n_G_decimal == 0:
                return float('nan'), {'error': 'G has no vertices'}
            
            denominator = n_G_decimal ** int(n_H_decimal)
            t_H_G = hom_count / denominator
            
            # Calculate edge density: average of all entries in G matrix
            total_sum = Decimal(0)
            total_entries = Decimal(self.n_G * self.n_G)
            
            for i in range(self.n_G):
                for j in range(self.n_G):
                    total_sum += Decimal(str(self.G_matrix[i][j]))
            
            p = total_sum / total_entries if total_entries > 0 else Decimal(0)
            
            # Calculate p^|E(H)|
            p_power_edges = p ** self.num_edges_H if self.num_edges_H > 0 else Decimal(1)
            
            # Sidorenko score = t_H_G - p^|E(H)|
            sidorenko_score = t_H_G - p_power_edges
            
            return float(sidorenko_score), {
                'hom_count': float(hom_count),
                't_H_G': float(t_H_G),
                'edge_density_p': float(p),
                'p_power_edges': float(p_power_edges),
                'num_edges_H': self.num_edges_H,
                'sidorenko_score': float(sidorenko_score)
            }
            
        except Exception as e:
            app.logger.error(f"Error in calculate_sidorenko_score: {str(e)}")
            app.logger.error(traceback.format_exc())
            return float('nan'), {'error': str(e)}

def vary_parameter(G_matrix, i, j, step_size):
    """
    Create a new G matrix with G[i][j] and G[j][i] varied by step_size.
    Maintains symmetry by updating both positions.
    Ensures values stay in [0, 1] range.
    """
    new_matrix = [row[:] for row in G_matrix]  # Deep copy
    new_value = G_matrix[i][j] + step_size
    
    # Clamp the value to [0, 1] range
    new_value = max(0.0, min(1.0, new_value))
    
    new_matrix[i][j] = new_value
    new_matrix[j][i] = new_value  # Maintain symmetry
    return new_matrix

def find_optimal_step(H_matrix, G_matrix, step_size=0.01, max_iterations=100):
    """
    Find the optimal step to decrease the Sidorenko score (looking for counterexamples).
    Uses the same range-based approach as parameter analysis to find the global minimum.
    Only checks upper triangle due to symmetry.
    """
    try:
        # Calculate current score
        calc = SidorenkoCalculator(H_matrix, G_matrix)
        current_score, _ = calc.calculate_sidorenko_score()
        
        if np.isnan(current_score):
            return None, "Current score calculation failed"
        
        best_score = current_score
        best_position = None
        best_new_value = None
        
        n = len(G_matrix)
        
        # Only check upper triangle (i <= j) due to symmetry
        for i in range(n):
            for j in range(i, n):  # j starts from i, not i+1, to include diagonal
                current_value = G_matrix[i][j]
                
                # Calculate how many steps we can take in each direction without going outside [0,1]
                max_steps_down = int(current_value / step_size) if step_size > 0 else 0
                max_steps_up = int((1 - current_value) / step_size) if step_size > 0 else 0
                
                # Test all values in the range [C-x*u, C+y*u] where values stay in [0,1]
                for step in range(-max_steps_down, max_steps_up + 1):
                    if step == 0:  # Skip current value
                        continue
                        
                    test_value = current_value + step * step_size
                    
                    # Ensure value stays in [0, 1]
                    test_value = max(0, min(1, test_value))
                    
                    # Create test matrix with this value
                    test_matrix = [row[:] for row in G_matrix]  # Deep copy
                    test_matrix[i][j] = test_value
                    test_matrix[j][i] = test_value  # Maintain symmetry
                    
                    # Calculate Sidorenko score
                    try:
                        calc_test = SidorenkoCalculator(H_matrix, test_matrix)
                        test_score, _ = calc_test.calculate_sidorenko_score()
                        
                        if not np.isnan(test_score) and test_score < best_score:
                            best_score = test_score
                            best_position = (i, j)
                            best_new_value = test_value
                            
                    except Exception as calc_error:
                        app.logger.warning(f"Calculation failed for position ({i},{j}) value {test_value}: {calc_error}")
                        continue  # Skip invalid values
        
        if best_position is not None:
            # Calculate the direction (difference from current value)
            current_val = G_matrix[best_position[0]][best_position[1]]
            direction = best_new_value - current_val
            improvement = current_score - best_score  # How much we improved (positive = better)
            
            return {
                'position': best_position,
                'direction': direction,
                'new_value': best_new_value,
                'improvement': -improvement,  # Return as negative for compatibility
                'decrease': improvement,
                'current_score': current_score,
                'best_score': best_score
            }, None
        else:
            return None, "No improvement found"
            
    except Exception as e:
        app.logger.error(f"Error in find_optimal_step: {str(e)}")
        app.logger.error(traceback.format_exc())
        return None, str(e)

@app.route('/')
def serve_frontend():
    return send_from_directory('frontend/build', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('frontend/build/static', path)

@app.route('/api/calculate', methods=['POST'])
def calculate_sidorenko():
    try:
        data = request.json
        H_matrix = data.get('H')
        G_matrix = data.get('G')
        
        if not H_matrix or not G_matrix:
            return jsonify({'error': 'Missing H or G matrix'}), 400
        
        # Validate that matrices are symmetric
        if not is_symmetric_matrix(H_matrix):
            return jsonify({'error': 'H matrix must be symmetric'}), 400
        if not is_symmetric_matrix(G_matrix):
            return jsonify({'error': 'G matrix must be symmetric'}), 400
        
        # Validate that G matrix has values in [0, 1] range
        if not is_valid_probability_matrix(G_matrix):
            return jsonify({'error': 'G matrix must have all values between 0 and 1 (inclusive)'}), 400
        
        calc = SidorenkoCalculator(H_matrix, G_matrix)
        score, details = calc.calculate_sidorenko_score()
        
        return jsonify({
            'score': score,
            'details': details
        })
        
    except Exception as e:
        app.logger.error(f"Error in calculate_sidorenko: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_plots_data', methods=['POST'])
def generate_plots_data():
    """
    Generate plot data for each matrix entry showing how varying it affects the Sidorenko score.
    Uses the correct parameter analysis: [C-xu, C-(x-1)u, ..., C, ..., C+yu] where values stay in [0,1]
    """
    try:
        data = request.json
        H_matrix = data.get('H')
        G_matrix = data.get('G')
        upsilon = data.get('step_size', 0.01)  # This is our 'u' parameter
        
        if not H_matrix or not G_matrix:
            return jsonify({'error': 'Missing H or G matrix'}), 400
        
        # Validate that matrices are symmetric
        if not is_symmetric_matrix(H_matrix):
            return jsonify({'error': 'H matrix must be symmetric'}), 400
        if not is_symmetric_matrix(G_matrix):
            return jsonify({'error': 'G matrix must be symmetric'}), 400
        
        # Validate that G matrix has values in [0, 1] range
        if not is_valid_probability_matrix(G_matrix):
            return jsonify({'error': 'G matrix must have all values between 0 and 1 (inclusive)'}), 400
        
        n = len(G_matrix)
        plots_data = {}
        
        # Generate plot data for each unique position (upper triangle including diagonal)
        for i in range(n):
            for j in range(i, n):  # Only upper triangle due to symmetry
                current_value = G_matrix[i][j]  # This is our 'C'
                
                # Calculate how many steps we can take in each direction without going outside [0,1]
                max_steps_down = int(current_value / upsilon) if upsilon > 0 else 0
                max_steps_up = int((1 - current_value) / upsilon) if upsilon > 0 else 0
                
                values = []
                scores = []
                
                # Generate values: [C-xu, C-(x-1)u, ..., C, ..., C+yu]
                # Go from -max_steps_down to +max_steps_up
                for step in range(-max_steps_down, max_steps_up + 1):
                    test_value = current_value + step * upsilon
                    
                    # Ensure value stays in [0, 1]
                    test_value = max(0, min(1, test_value))
                    
                    # Create test matrix with this value
                    test_matrix = [row[:] for row in G_matrix]  # Deep copy
                    test_matrix[i][j] = test_value
                    test_matrix[j][i] = test_value  # Maintain symmetry
                    
                    # Calculate Sidorenko score
                    try:
                        calc = SidorenkoCalculator(H_matrix, test_matrix)
                        score, _ = calc.calculate_sidorenko_score()
                        
                        if not np.isnan(score):
                            values.append(test_value)
                            scores.append(score)
                    except Exception as calc_error:
                        app.logger.warning(f"Calculation failed for position ({i},{j}) value {test_value}: {calc_error}")
                        continue  # Skip invalid values
                
                if values:  # Only include if we have valid data
                    plots_data[f"{i},{j}"] = {
                        'position': [i, j],
                        'current_value': current_value,
                        'values': values,
                        'scores': scores,
                        'title': f'G[{i},{j}]' if i == j else f'G[{i},{j}] = G[{j},{i}]',
                        'upsilon': upsilon,
                        'steps_down': max_steps_down,
                        'steps_up': max_steps_up
                    }
        
        return jsonify({
            'plots_data': plots_data,
            'matrix_size': n,
            'upsilon': upsilon
        })
        
    except Exception as e:
        app.logger.error(f"Error in generate_plots_data: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/optimize_step', methods=['POST'])
def optimize_step():
    try:
        data = request.json
        H_matrix = data.get('H')
        G_matrix = data.get('G')
        step_size = data.get('step_size', 0.01)
        
        if not H_matrix or not G_matrix:
            return jsonify({'error': 'Missing H or G matrix'}), 400
        
        # Validate that matrices are symmetric
        if not is_symmetric_matrix(H_matrix):
            return jsonify({'error': 'H matrix must be symmetric'}), 400
        if not is_symmetric_matrix(G_matrix):
            return jsonify({'error': 'G matrix must be symmetric'}), 400
        
        # Validate that G matrix has values in [0, 1] range
        if not is_valid_probability_matrix(G_matrix):
            return jsonify({'error': 'G matrix must have all values between 0 and 1 (inclusive)'}), 400
        
        result, error = find_optimal_step(H_matrix, G_matrix, step_size)
        
        if error:
            return jsonify({'error': error}), 400
        
        if result:
            # Apply the optimal value
            i, j = result['position']
            new_G_matrix = [row[:] for row in G_matrix]  # Deep copy
            new_G_matrix[i][j] = result['new_value']
            new_G_matrix[j][i] = result['new_value']  # Maintain symmetry
            
            # Calculate new score
            calc = SidorenkoCalculator(H_matrix, new_G_matrix)
            new_score, details = calc.calculate_sidorenko_score()
            
            return jsonify({
                'success': True,
                'new_matrix': new_G_matrix,
                'new_score': new_score,
                'details': details,
                'optimization_info': {
                    'position': result['position'],
                    'direction': result['direction'],
                    'improvement': result['improvement'],
                    'step_size': step_size,
                    'new_value': result['new_value'],
                    'old_value': G_matrix[result['position'][0]][result['position'][1]]
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No improvement found'
            })
        
    except Exception as e:
        app.logger.error(f"Error in optimize_step: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)
