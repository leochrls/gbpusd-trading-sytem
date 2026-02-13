import { useState, type FormEvent } from 'react';
import { Card } from './ui/Card';
import type { PredictionRequest } from '../api';
import { Settings2, RefreshCcw, PlayCircle } from 'lucide-react';

interface PredictionFormProps {
    initialValues: Record<string, number>;
    onSubmit: (data: PredictionRequest) => void;
    isLoading: boolean;
}

const FEATURE_GROUPS = {
    'Short Term': [
        'return_1', 'return_4', 'ema_diff',
        'rsi_14', 'rolling_std_20', 'range_15m',
        'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio'
    ],
    'Regime': [
        'distance_to_ema200', 'slope_ema50', 'atr_14',
        'rolling_std_100', 'volatility_ratio', 'adx_14',
        'macd', 'macd_signal', 'macd_histogram'
    ]
};

export function PredictionForm({ initialValues, onSubmit, isLoading }: PredictionFormProps) {
    const [features, setFeatures] = useState<Record<string, number>>(initialValues);
    const [position, setPosition] = useState<number>(0);
    const [pnl, setPnl] = useState<number>(0);
    const [drawdown, setDrawdown] = useState<number>(0);

    const handleSubmit = (e: FormEvent) => {
        e.preventDefault();
        onSubmit({
            features,
            position,
            pnl_unrealized: pnl,
            drawdown
        });
    };

    const handleReset = () => {
        setFeatures(initialValues);
        setPosition(0);
        setPnl(0);
        setDrawdown(0);
    };

    const FeatureInput = ({ name }: { name: string }) => (
        <div className="flex flex-col gap-1">
            <label className="text-[10px] uppercase tracking-wider text-text-secondary font-medium truncate" title={name}>
                {name.replace(/_/g, ' ')}
            </label>
            <input
                type="number"
                step="any"
                value={features[name] ?? 0}
                onChange={(e) => setFeatures({ ...features, [name]: parseFloat(e.target.value) || 0 })}
                className="bg-input border border-border rounded-md px-2 py-1 text-sm text-text-primary font-mono focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent transition-all"
            />
        </div>
    );

    return (
        <Card
            title="Prediction Parameters"
            className="h-full"
            action={
                <button
                    onClick={handleReset}
                    className="text-text-secondary hover:text-text-primary transition-colors p-1"
                    title="Reset to defaults"
                >
                    <RefreshCcw className="w-4 h-4" />
                </button>
            }
        >
            <form onSubmit={handleSubmit} className="flex flex-col h-full">
                <div className="flex-1 overflow-y-auto pr-2 space-y-6 custom-scrollbar">

                    {/* Feature Groups */}
                    {Object.entries(FEATURE_GROUPS).map(([group, keys]) => (
                        <div key={group}>
                            <div className="flex items-center gap-2 mb-3 pb-1 border-b border-border/50">
                                <Settings2 className="w-3 h-3 text-accent" />
                                <h4 className="text-xs font-semibold uppercase tracking-wider text-text-secondary">{group}</h4>
                            </div>
                            <div className="grid grid-cols-3 gap-3">
                                {keys.map(key => <FeatureInput key={key} name={key} />)}
                            </div>
                        </div>
                    ))}

                    {/* Position & PnL Section */}
                    <div>
                        <div className="flex items-center gap-2 mb-3 pb-1 border-b border-border/50">
                            <h4 className="text-xs font-semibold uppercase tracking-wider text-text-secondary">Position Context</h4>
                        </div>
                        <div className="grid grid-cols-3 gap-3">
                            <div className="flex flex-col gap-1">
                                <label className="text-[10px] uppercase text-text-secondary font-medium">Current Position</label>
                                <select
                                    value={position}
                                    onChange={(e) => setPosition(parseInt(e.target.value))}
                                    className="bg-input border border-border rounded-md px-2 py-1.5 text-sm text-text-primary focus:outline-none focus:border-accent"
                                >
                                    <option value={1}>LONG (1)</option>
                                    <option value={0}>FLAT (0)</option>
                                    <option value={-1}>SHORT (-1)</option>
                                </select>
                            </div>
                            <div className="flex flex-col gap-1">
                                <label className="text-[10px] uppercase text-text-secondary font-medium">PnL Unrealized (%)</label>
                                <input
                                    type="number"
                                    step="0.01"
                                    value={pnl}
                                    onChange={(e) => setPnl(parseFloat(e.target.value))}
                                    className="bg-input border border-border rounded-md px-2 py-1 text-sm text-text-primary font-mono"
                                />
                            </div>
                            <div className="flex flex-col gap-1">
                                <label className="text-[10px] uppercase text-text-secondary font-medium">Drawdown (%)</label>
                                <input
                                    type="number"
                                    step="0.01"
                                    value={drawdown}
                                    onChange={(e) => setDrawdown(parseFloat(e.target.value))}
                                    className="bg-input border border-border rounded-md px-2 py-1 text-sm text-text-primary font-mono"
                                />
                            </div>
                        </div>
                    </div>
                </div>

                <div className="mt-6 pt-4 border-t border-border">
                    <button
                        type="submit"
                        disabled={isLoading}
                        className="w-full bg-accent hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-3 px-4 rounded-lg shadow-lg shadow-blue-500/20 active:scale-[0.99] transition-all flex items-center justify-center gap-2"
                    >
                        {isLoading ? (
                            <>
                                <RefreshCcw className="w-5 h-5 animate-spin" />
                                Predicting...
                            </>
                        ) : (
                            <>
                                <PlayCircle className="w-5 h-5" />
                                RUN PREDICTION
                            </>
                        )}
                    </button>
                </div>
            </form>
        </Card>
    );
}
