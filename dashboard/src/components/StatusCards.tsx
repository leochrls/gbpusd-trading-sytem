import { Activity, Server, Database } from 'lucide-react';
import { Card } from './ui/Card';
import { Badge } from './ui/Badge';

interface StatusCardsProps {
    apiHealth: boolean;
    modelLoaded: boolean;
    activeModel: { name: string; version: string; type: string } | null;
    totalRequests: number;
    lastUpdated: string;
}

export function StatusCards({ apiHealth, modelLoaded, activeModel, totalRequests, lastUpdated }: StatusCardsProps) {
    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            {/* API Status Card */}
            <Card>
                <div className="flex items-center justify-between mb-2">
                    <span className="text-text-secondary text-sm font-medium">System Status</span>
                    <Activity className={`w-4 h-4 ${apiHealth ? 'text-signal-buy' : 'text-signal-sell'}`} />
                </div>
                <div className="flex items-end justify-between">
                    <div>
                        <div className="flex items-center gap-2 mb-1">
                            <div className={`w-2 h-2 rounded-full ${apiHealth ? 'bg-signal-buy' : 'bg-signal-sell'} animate-pulse`} />
                            <span className="text-xl font-bold text-text-primary">
                                {apiHealth ? 'Online' : 'Offline'}
                            </span>
                        </div>
                        <span className="text-xs text-text-secondary block">
                            Model Loaded: <span className={modelLoaded ? 'text-signal-buy' : 'text-signal-sell'}>{modelLoaded ? 'Yes' : 'No'}</span>
                        </span>
                    </div>
                </div>
            </Card>

            {/* Active Model Card */}
            <Card>
                <div className="flex items-center justify-between mb-2">
                    <span className="text-text-secondary text-sm font-medium">Active Model</span>
                    <Database className="w-4 h-4 text-accent" />
                </div>
                {activeModel ? (
                    <div className="flex flex-col gap-1">
                        <div className="flex items-center gap-2">
                            <span className="text-xl font-bold text-text-primary truncate" title={activeModel.name}>
                                {activeModel.name}
                            </span>
                            <Badge label={activeModel.type} variant={activeModel.type === 'RL' ? 'rl' : 'ml'} className="ml-auto" />
                        </div>
                        <span className="text-xs text-text-secondary font-mono">v{activeModel.version}</span>
                    </div>
                ) : (
                    <div className="flex flex-col gap-1">
                        <span className="text-xl font-bold text-text-secondary">No Model</span>
                        <span className="text-xs text-text-secondary">Waiting for initialization...</span>
                    </div>
                )}
            </Card>

            {/* Total Requests Card */}
            <Card>
                <div className="flex items-center justify-between mb-2">
                    <span className="text-text-secondary text-sm font-medium">Total Predictions</span>
                    <Server className="w-4 h-4 text-accent" />
                </div>
                <div className="flex items-end justify-between">
                    <span className="text-3xl font-mono font-bold text-text-primary tracking-tight">
                        {totalRequests.toLocaleString()}
                    </span>
                    <span className="text-xs text-text-secondary mb-1">
                        Last: {new Date(lastUpdated).toLocaleTimeString()}
                    </span>
                </div>
            </Card>
        </div>
    );
}
