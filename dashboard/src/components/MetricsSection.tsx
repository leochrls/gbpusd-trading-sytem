import { Card } from './ui/Card';
import { Badge } from './ui/Badge';
import type { MetricsResponse, RequiredFeaturesResponse, AvailableModelsResponse } from '../api';
import { FileText, Layers } from 'lucide-react';

interface MetricsSectionProps {
    metrics: MetricsResponse | null;
    features: RequiredFeaturesResponse | null;
    availableModels: AvailableModelsResponse | null;
}

export function MetricsSection({ metrics, features, availableModels }: MetricsSectionProps) {
    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            {/* 1. Model Metrics */}
            <Card title="Latest Model Metrics">
                {!metrics ? (
                    <div className="text-sm text-text-secondary text-center py-4">No metrics available</div>
                ) : (
                    <div className="space-y-3">
                        <div className="flex justify-between items-center border-b border-border/50 pb-2">
                            <span className="text-sm text-text-secondary">Model</span>
                            <span className="text-sm font-medium text-text-primary">{metrics.model_name} (v{metrics.model_version})</span>
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                            {Object.entries(metrics.metrics).map(([key, value]) => (
                                <div key={key} className="flex flex-col">
                                    <span className="text-xs uppercase text-text-secondary">{key.replace(/_/g, ' ')}</span>
                                    <span className="font-mono text-sm text-text-primary">
                                        {typeof value === 'number' ? value.toFixed(4) : String(value)}
                                    </span>
                                </div>
                            ))}
                        </div>
                        <div className="pt-2 text-[10px] text-text-secondary text-right">
                            Updated: {new Date(metrics.timestamp).toLocaleString()}
                        </div>
                    </div>
                )}
            </Card>

            {/* 2. Required Features */}
            <Card title="Required Features">
                <div className="flex flex-wrap gap-2 max-h-[200px] overflow-y-auto custom-scrollbar">
                    {!features ? (
                        <div className="text-sm text-text-secondary">Loading features...</div>
                    ) : (
                        features.features.map(feature => (
                            <span key={feature} className="px-2 py-1 bg-input rounded text-xs text-text-secondary border border-border/50 font-mono">
                                {feature}
                            </span>
                        ))
                    )}
                </div>
                {features && (
                    <div className="mt-3 text-xs text-text-secondary flex items-center gap-1">
                        <Layers className="w-3 h-3" />
                        {features.count} features tracked
                    </div>
                )}
            </Card>

            {/* 3. Available Models */}
            <Card title="Available Models">
                <div className="space-y-4 max-h-[200px] overflow-y-auto custom-scrollbar">
                    {!availableModels ? (
                        <div className="text-sm text-text-secondary">Loading models...</div>
                    ) : (
                        Object.entries(availableModels.available).map(([version, types]) => (
                            <div key={version} className="bg-input/30 rounded-lg p-3 border border-border/30">
                                <div className="flex items-center gap-2 mb-2">
                                    <FileText className="w-4 h-4 text-accent" />
                                    <span className="text-sm font-semibold text-text-primary">Version {version}</span>
                                </div>
                                <div className="space-y-2 pl-6">
                                    {types.ml.length > 0 && (
                                        <div className="flex flex-col gap-1">
                                            <span className="text-[10px] text-text-secondary uppercase">Machine Learning</span>
                                            <div className="flex flex-wrap gap-2">
                                                {types.ml.map(model => (
                                                    <Badge key={model} label={model} variant="ml" />
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                    {types.rl.length > 0 && (
                                        <div className="flex flex-col gap-1">
                                            <span className="text-[10px] text-text-secondary uppercase">Reinforcement Learning</span>
                                            <div className="flex flex-wrap gap-2">
                                                {types.rl.map(model => (
                                                    <Badge key={model} label={model} variant="rl" />
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </Card>
        </div>
    );
}
