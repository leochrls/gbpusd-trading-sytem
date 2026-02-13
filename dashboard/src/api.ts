const API_BASE_URL = 'http://localhost:8000';

export interface HealthResponse {
    status: string;
    model_loaded: boolean;
    model_name: string | null;
    model_type: string | null;
    uptime_seconds: number;
    total_requests: number;
    timestamp: string;
}

export interface PredictionRequest {
    features: Record<string, number>;
    position: number;
    pnl_unrealized: number;
    drawdown: number;
}

export interface PredictionResponse {
    action: "BUY" | "SELL" | "HOLD";
    confidence: number;
    model_used: string;
    model_type: string;
    model_version: string;
    timestamp: string;
    details: any;
}

export interface AvailableModelsResponse {
    available: {
        [version: string]: {
            ml: string[];
            rl: string[];
        };
    };
    current: {
        name: string;
        version: string;
        type: string;
        loaded_at: string;
    } | null;
}

export interface MetricsResponse {
    model_name: string;
    model_type: string;
    model_version: string;
    metrics: Record<string, any>;
    timestamp: string;
}

export interface RequiredFeaturesResponse {
    features: string[];
    count: number;
    description: string;
}

export const api = {
    async getHealth(): Promise<HealthResponse> {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) throw new Error('API unreachable');
        return response.json();
    },

    async predict(data: PredictionRequest): Promise<PredictionResponse> {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        if (!response.ok) throw new Error('Prediction failed');
        return response.json();
    },

    async getAvailableModels(): Promise<AvailableModelsResponse> {
        const response = await fetch(`${API_BASE_URL}/models/available`);
        if (!response.ok) throw new Error('Failed to fetch models');
        return response.json();
    },

    async getLatestMetrics(): Promise<MetricsResponse> {
        const response = await fetch(`${API_BASE_URL}/metrics/latest`);
        if (!response.ok) throw new Error('Failed to fetch metrics');
        return response.json();
    },

    async getRequiredFeatures(): Promise<RequiredFeaturesResponse> {
        const response = await fetch(`${API_BASE_URL}/features/required`);
        if (!response.ok) throw new Error('Failed to fetch required features');
        return response.json();
    }
};
