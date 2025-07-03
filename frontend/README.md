# Sidorenko's Conjecture Visualization Tool - Frontend

This is a React frontend for visualizing and exploring Sidorenko's conjecture through interactive plots and optimization.

## Environment Configuration

The application uses environment variables for configuration. Copy `.env.example` to `.env` and configure your settings:

```bash
cp .env.example .env
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REACT_APP_API_BASE_URL` | Base URL for the backend API | `http://localhost:5002` |
| `REACT_APP_ENVIRONMENT` | Application environment | `development` |

### Configuration Files

- **`.env`** - Your local environment configuration (do not commit)
- **`.env.example`** - Template showing required variables
- **`src/config.ts`** - Centralized configuration management

### Different Environments

#### Development (default)
```env
REACT_APP_API_BASE_URL=http://localhost:5002
REACT_APP_ENVIRONMENT=development
```

#### Production
```env
REACT_APP_API_BASE_URL=https://your-api-domain.com
REACT_APP_ENVIRONMENT=production
```

## Getting Started

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Start development server:**
   ```bash
   npm start
   ```

4. **Open in browser:**
   Navigate to [http://localhost:3000](http://localhost:3000)

## Available Scripts

- `npm start` - Start development server
- `npm build` - Build for production
- `npm test` - Run tests
- `npm run eject` - Eject from Create React App

## Features

- **Interactive Matrix Input** - Edit pattern and host graphs
- **Real-time Calculations** - Compute Sidorenko scores
- **Optimization Tools** - Find potential counterexamples
- **Data Visualization** - Charts showing score evolution and optimization progress
- **Matrix Heatmaps** - Visual representation of graph structures

## API Integration

The frontend communicates with a Flask backend API. Make sure the backend is running on the configured URL (default: `http://localhost:5002`) before using the application.

### API Endpoints Used

- `POST /api/calculate` - Calculate Sidorenko score
- `POST /api/optimize_step` - Perform optimization step

## Configuration Management

The app uses a centralized configuration system in `src/config.ts` that:

- Provides type-safe access to environment variables
- Includes helper functions for building API URLs
- Supports different environments (development, production)
- Provides fallback values for all configurations

## Troubleshooting

### Backend Connection Issues

If you see connection errors:

1. Check that the backend is running on the configured port
2. Verify the `REACT_APP_API_BASE_URL` in your `.env` file
3. Ensure CORS is properly configured on the backend

### Environment Variables Not Loading

- Environment variables must start with `REACT_APP_`
- Restart the development server after changing `.env`
- Check that `.env` is in the frontend root directory
