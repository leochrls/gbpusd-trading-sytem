import { useState, useEffect, useCallback } from 'react';
import { Header } from '../components/Header';
import { StatusCards } from '../components/StatusCards';
import { PredictionForm } from '../components/PredictionForm';
import { ResultPanel } from '../components/ResultPanel';
import { AnalyticsCharts } from '../components/AnalyticsCharts';
import { MetricsSection } from '../components/MetricsSection';
import { ToastContainer, type ToastMessage } from '../components/ui/Toast';
import { api, type HealthResponse, type AvailableModelsResponse, type MetricsResponse, type RequiredFeaturesResponse, type PredictionResponse, type PredictionRequest } from '../api';

// Initial default features as specified in prompt
const DEFAULT_FEATURES = {
    return_1: 0.0012, return_4: -0.0005, ema_diff: 0.0003, rsi_14: 52.3,
    rolling_std_20: 0.0008, range_15m: 0.0025, body_ratio: 0.6,
    upper_wick_ratio: 0.2, lower_wick_ratio: 0.2,
    distance_to_ema200: 0.005, slope_ema50: 0.0001, atr_14: 0.0020,
    rolling_std_100: 0.0007, volatility_ratio: 1.14, adx_14: 25.3,
    macd: 0.0002, macd_signal: 0.0001, macd_histogram: 0.0001
};

import { AnalysisPanel } from '../components/AnalysisPanel';

export function Dashboard() {
    const [activeTab, setActiveTab] = useState<'overview' | 'analysis'>('overview');
    const [health, setHealth] = useState<HealthResponse | null>(null);
    const [availableModels, setAvailableModels] = useState<AvailableModelsResponse | null>(null);
    const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
    const [requiredFeatures, setRequiredFeatures] = useState<RequiredFeaturesResponse | null>(null);
    const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
    const [history, setHistory] = useState<PredictionResponse[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [toasts, setToasts] = useState<ToastMessage[]>([]);

    const addToast = (type: 'success' | 'error', message: string) => {
        const id = Math.random().toString(36).substring(7);
        setToasts(prev => [...prev, { id, type, message }]);
    };

    const removeToast = (id: string) => {
        setToasts(prev => prev.filter(t => t.id !== id));
    };

    const fetchAllData = useCallback(async () => {
        try {
            const [healthData, modelsData, metricsData, featuresData] = await Promise.all([
                api.getHealth(),
                api.getAvailableModels(),
                api.getLatestMetrics(),
                api.getRequiredFeatures()
            ]);

            setHealth(healthData);
            setAvailableModels(modelsData);
            setMetrics(metricsData);
            setRequiredFeatures(featuresData);
        } catch (err) {
            console.error(err);
            setHealth(prev => prev ? { ...prev, status: 'offline' } : null);
        }
    }, []);

    useEffect(() => {
        fetchAllData();
        const interval = setInterval(() => {
            api.getHealth().then(setHealth).catch(() => setHealth(null));
        }, 30000);
        return () => clearInterval(interval);
    }, [fetchAllData]);

    const handleManualRefresh = async () => {
        setIsLoading(true);
        await fetchAllData();
        setIsLoading(false);
        addToast('success', 'Dashboard data refreshed');
    };

    const handlePredict = async (data: PredictionRequest) => {
        setIsLoading(true);
        try {
            const result = await api.predict(data);
            setPrediction(result);
            setHistory(prev => [result, ...prev].slice(0, 10));
            addToast('success', `Prediction: ${result.action}`);

            api.getHealth().then(setHealth);
            api.getLatestMetrics().then(setMetrics);
        } catch (err) {
            addToast('error', 'Prediction failed. Check API connection.');
        } finally {
            setIsLoading(false);
        }
    };

    const isConnected = health?.status === 'online' || (health?.uptime_seconds !== undefined);

    return (
        <div className="min-h-screen bg-background text-text-primary font-sans selection:bg-accent/30 flex flex-col">
            <Header
                isConnected={isConnected}
                uptime={
                    health ? new Date(health.uptime_seconds * 1000).toISOString().substr(11, 8) : '00:00:00'
                }
                totalRequests={health?.total_requests || 0}
                activeModel={health?.model_name ? { name: health.model_name, type: health.model_type || 'Unknown' } : undefined}
                onRefresh={handleManualRefresh}
                isRefreshing={isLoading}
            />

            {!isConnected && (
                <div className="bg-red-500/10 border-b border-red-500/20 text-red-500 px-6 py-2 text-center text-sm font-medium animate-in slide-in-from-top">
                    API is unreachable — Please check if the backend service is running at http://localhost:8000
                </div>
            )}

            <main className="flex-1 p-6 container mx-auto max-w-7xl">
                <StatusCards
                    apiHealth={isConnected}
                    modelLoaded={health?.model_loaded || false}
                    activeModel={availableModels?.current || null}
                    totalRequests={health?.total_requests || 0}
                    lastUpdated={new Date().toISOString()}
                />

                {/* Tab Navigation */}
                <div className="flex items-center gap-4 mb-6 border-b border-border">
                    <button
                        onClick={() => setActiveTab('overview')}
                        className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${activeTab === 'overview'
                            ? 'border-accent text-accent'
                            : 'border-transparent text-text-secondary hover:text-text-primary'
                            }`}
                    >
                        Overview
                    </button>
                    <button
                        onClick={() => setActiveTab('analysis')}
                        className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${activeTab === 'analysis'
                            ? 'border-accent text-accent'
                            : 'border-transparent text-text-secondary hover:text-text-primary'
                            }`}
                    >
                        Market Analysis
                    </button>
                </div>

                {activeTab === 'overview' ? (
                    <div className="animate-in fade-in slide-in-from-left-4 duration-300">
                        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 mb-6">
                            <div className="lg:col-span-3 h-[600px]">
                                <PredictionForm
                                    initialValues={DEFAULT_FEATURES}
                                    onSubmit={handlePredict}
                                    isLoading={isLoading}
                                />
                            </div>
                            <div className="lg:col-span-2 h-[600px]">
                                <ResultPanel
                                    prediction={prediction}
                                    history={history}
                                />
                            </div>
                        </div>

                        <AnalyticsCharts history={history} />

                        <MetricsSection
                            metrics={metrics}
                            features={requiredFeatures}
                            availableModels={availableModels}
                        />
                    </div>
                ) : (
                    <AnalysisPanel />
                )}
            </main>

            <ToastContainer toasts={toasts} onDismiss={removeToast} />
        </div>
    );
}
