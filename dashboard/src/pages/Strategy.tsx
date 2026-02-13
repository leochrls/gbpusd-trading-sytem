import { Brain, Network, Database, Zap } from 'lucide-react';


export function Strategy() {
    return (
        <div className="container mx-auto px-6 py-12">
            <div className="text-center mb-20">
                <h1 className="text-4xl font-bold mb-4">The Methodology</h1>
                <p className="text-text-secondary max-w-2xl mx-auto">
                    Combining traditional quantitative finance with state-of-the-art Deep Reinforcement Learning.
                </p>
            </div>

            {/* Process Flow */}
            <div className="relative">
                <div className="hidden md:block absolute top-[100px] left-1/4 right-1/4 h-0.5 bg-gradient-to-r from-blue-500/0 via-blue-500/50 to-blue-500/0" />

                <div className="grid grid-cols-1 md:grid-cols-3 gap-12 relative z-10">
                    <ProcessStep
                        icon={Database}
                        step="01"
                        title="Data Ingestion"
                        description="Our pipelines ingest 18+ market indicators including price action, momentum (RSI, MACD), and volatility metrics (ATR, Bollinger Bands) in real-time."
                    />
                    <ProcessStep
                        icon={Brain}
                        step="02"
                        title="Ensemble Inference"
                        description="Multiple ML models (XGBoost, LSTM) generate feature vectors which are fed into our central Reinforcement Learning agent (PPO/DQN)."
                    />
                    <ProcessStep
                        icon={Zap}
                        step="03"
                        title="Execution"
                        description="The agent outputs discrete actions (BUY/SELL/HOLD) with confidence scores. Orders are routed via low-latency API with slippage protection."
                    />
                </div>
            </div>

            <div className="mt-32 grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
                <div>
                    <h2 className="text-3xl font-bold mb-6">Why Reinforcement Learning?</h2>
                    <div className="space-y-6">
                        <Benefit
                            title="Adaptability"
                            text="Unlike static heuristic strategies, RL agents learn from market feedback, adapting to regime changes (e.g. trending vs ranging)."
                        />
                        <Benefit
                            title="Multi-Objective Optimization"
                            text="Our agents aim to maximize Profit & Loss while simultaneously minimizing Drawdown and Transaction Costs."
                        />
                        <Benefit
                            title="Latency Arbitrage"
                            text="Optimized C++ backends allow for decision making in microseconds, capturing opportunities before human traders can react."
                        />
                    </div>
                </div>
                <div className="bg-card border border-border rounded-2xl p-8 aspect-square flex items-center justify-center relative overflow-hidden">
                    {/* Abstract Neural Network Visualization */}
                    <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-blue-900/20 via-background to-background" />
                    <Network className="w-64 h-64 text-accent/20 animate-pulse" />
                    <div className="absolute inset-0 flex items-center justify-center">
                        <span className="text-xl font-mono text-accent font-bold">Policy Network</span>
                    </div>
                </div>
            </div>
        </div>
    );
}

function ProcessStep({ icon: Icon, step, title, description }: { icon: any, step: string, title: string, description: string }) {
    return (
        <div className="bg-card border border-border p-8 rounded-xl relative group hover:border-accent/50 transition-colors">
            <div className="text-6xl font-black text-input absolute -top-8 right-4 select-none group-hover:text-accent/10 transition-colors">{step}</div>
            <div className="w-14 h-14 bg-accent/10 rounded-xl flex items-center justify-center mb-6">
                <Icon className="w-7 h-7 text-accent" />
            </div>
            <h3 className="text-xl font-bold mb-3">{title}</h3>
            <p className="text-text-secondary text-sm leading-relaxed">{description}</p>
        </div>
    );
}

function Benefit({ title, text }: { title: string, text: string }) {
    return (
        <div className="flex gap-4">
            <div className="w-1.5 h-full min-h-[50px] bg-accent/50 rounded-full" />
            <div>
                <h4 className="font-bold text-lg mb-1">{title}</h4>
                <p className="text-text-secondary text-sm">{text}</p>
            </div>
        </div>
    );
}
