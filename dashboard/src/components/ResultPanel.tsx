import { useState } from 'react';
import { Card } from './ui/Card';
import { Badge } from './ui/Badge';
import type { PredictionResponse } from '../api';
import { ChevronDown, ChevronUp, AlertTriangle } from 'lucide-react';
import { clsx } from 'clsx';

interface ResultPanelProps {
    prediction: PredictionResponse | null;
    history: PredictionResponse[];
}

export function ResultPanel({ prediction, history }: ResultPanelProps) {
    const [showDetails, setShowDetails] = useState(false);

    const getActionColor = (action: string) => {
        switch (action) {
            case 'BUY': return 'text-signal-buy';
            case 'SELL': return 'text-signal-sell';
            default: return 'text-signal-hold';
        }
    };

    const getConfidenceColor = (conf: number) => {
        if (conf >= 0.7) return 'bg-green-500';
        if (conf >= 0.5) return 'bg-yellow-500';
        return 'bg-red-500';
    };

    return (
        <div className="flex flex-col gap-6 h-full">
            {/* Latest Prediction Result */}
            <Card className="flex-1 min-h-[300px] flex flex-col justify-center relative overflow-hidden">
                {!prediction ? (
                    <div className="flex flex-col items-center justify-center h-full text-text-secondary opacity-50">
                        <div className="w-16 h-16 rounded-full border-4 border-dashed border-text-secondary animate-[spin_10s_linear_infinite] mb-4" />
                        <p>Ready to predict</p>
                    </div>
                ) : (
                    <div className="flex flex-col items-center z-10">
                        <span className="text-sm font-medium text-text-secondary uppercase tracking-widest mb-2">Recommended Action</span>
                        <h2 className={clsx("text-6xl font-black tracking-tighter mb-6 drop-shadow-lg", getActionColor(prediction.action))}>
                            {prediction.action}
                        </h2>

                        <div className="w-full max-w-xs mb-2">
                            <div className="flex justify-between text-xs font-medium text-text-secondary mb-1">
                                <span>Confidence</span>
                                <span>{(prediction.confidence * 100).toFixed(1)}%</span>
                            </div>
                            <div className="h-2 bg-card border border-border rounded-full overflow-hidden">
                                <div
                                    className={clsx("h-full transition-all duration-500 ease-out", getConfidenceColor(prediction.confidence))}
                                    style={{ width: `${prediction.confidence * 100}%` }}
                                />
                            </div>
                        </div>

                        {prediction.confidence < 0.5 && (
                            <div className="flex items-center gap-2 text-yellow-500 text-xs font-medium bg-yellow-500/10 px-3 py-1 rounded-full border border-yellow-500/20 mb-6">
                                <AlertTriangle className="w-3 h-3" />
                                Low confidence - High uncertainty
                            </div>
                        )}

                        <div className="text-center mb-6">
                            <Badge label={prediction.model_used} variant="default" className="mr-2" />
                            <Badge label={prediction.model_type} variant={prediction.model_type === 'RL' ? 'rl' : 'ml'} />
                        </div>

                        <button
                            onClick={() => setShowDetails(!showDetails)}
                            className="flex items-center gap-1 text-xs text-text-secondary hover:text-text-primary transition-colors"
                        >
                            {showDetails ? 'Hide Details' : 'Show Model Details'}
                            {showDetails ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                        </button>

                        {showDetails && (
                            <div className="mt-4 w-full bg-input rounded-lg p-3 text-xs font-mono text-text-secondary border border-border animate-in slide-in-from-top-2 fade-in">
                                <pre>{JSON.stringify(prediction.details, null, 2)}</pre>
                            </div>
                        )}
                    </div>
                )}

                {/* Background Glow Effect */}
                {prediction && (
                    <div className={clsx(
                        "absolute inset-0 opacity-10 blur-3xl transition-colors duration-500 pointer-events-none",
                        prediction.action === 'BUY' ? 'bg-signal-buy' : prediction.action === 'SELL' ? 'bg-signal-sell' : 'bg-signal-hold'
                    )} />
                )}
            </Card>

            {/* Prediction History Table */}
            <Card title="Recent History" className="flex-1 overflow-hidden flex flex-col">
                <div className="overflow-auto custom-scrollbar flex-1">
                    <table className="w-full text-sm text-left">
                        <thead className="text-xs text-text-secondary uppercase bg-input/50 sticky top-0">
                            <tr>
                                <th className="px-3 py-2">#</th>
                                <th className="px-3 py-2">Signal</th>
                                <th className="px-3 py-2">Conf.</th>
                                <th className="px-3 py-2">Model</th>
                                <th className="px-3 py-2 text-right">Time</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-border/50">
                            {history.length === 0 ? (
                                <tr>
                                    <td colSpan={5} className="px-3 py-8 text-center text-text-secondary italic">
                                        No predictions yet
                                    </td>
                                </tr>
                            ) : (
                                history.slice().reverse().map((item, idx) => (
                                    <tr key={idx} className="hover:bg-input/30 transition-colors">
                                        <td className="px-3 py-2 font-mono text-text-secondary">{history.length - idx}</td>
                                        <td className="px-3 py-2">
                                            <Badge
                                                label={item.action}
                                                variant={item.action.toLowerCase() as any}
                                                className="scale-90 origin-left"
                                            />
                                        </td>
                                        <td className="px-3 py-2 font-mono">{(item.confidence * 100).toFixed(1)}%</td>
                                        <td className="px-3 py-2 truncate max-w-[100px] text-xs" title={item.model_used}>{item.model_used}</td>
                                        <td className="px-3 py-2 text-right text-xs text-text-secondary font-mono">
                                            {new Date(item.timestamp).toLocaleTimeString()}
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </Card>
        </div>
    );
}
