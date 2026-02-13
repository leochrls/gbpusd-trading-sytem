import { motion } from 'framer-motion';
import { ArrowRight, TrendingUp, Shield, Cpu, ChevronRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import { ComparisonChart } from '../components/ComparisonChart';

export function Landing() {
    return (
        <div className="flex flex-col">
            {/* Hero Section */}
            <section className="relative px-6 pt-20 pb-32 overflow-hidden">
                <div className="absolute inset-0 bg-background">
                    <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-accent/20 rounded-full blur-[120px] -translate-y-1/2 translate-x-1/2" />
                    <div className="absolute bottom-0 left-0 w-[300px] h-[300px] bg-purple-500/10 rounded-full blur-[100px]" />
                </div>

                <div className="container mx-auto max-w-5xl relative z-10 text-center">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6 }}
                    >
                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent/10 border border-accent/20 text-accent text-sm font-medium mb-8">
                            <span className="relative flex h-2 w-2">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-2 w-2 bg-accent"></span>
                            </span>
                            Algorithm V2.4 Live
                        </div>

                        <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-8 bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
                            Outperform the Market with <br />
                            <span className="text-white">Reinforcement Learning</span>
                        </h1>

                        <p className="text-xl text-text-secondary max-w-2xl mx-auto mb-12">
                            Our autonomous trading system adapts to live GBP/USD volatility in real-time,
                            delivering institutional-grade performance without the emotional bias.
                        </p>

                        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                            <Link
                                to="/contact"
                                className="px-8 py-4 bg-accent hover:bg-blue-600 text-white font-semibold rounded-xl shadow-lg shadow-blue-500/25 transition-all text-lg flex items-center gap-2"
                            >
                                Request Access
                                <ArrowRight className="w-5 h-5" />
                            </Link>
                            <Link
                                to="/performance"
                                className="px-8 py-4 bg-card hover:bg-input border border-border text-text-primary font-medium rounded-xl transition-all text-lg"
                            >
                                View Performance
                            </Link>
                        </div>
                    </motion.div>
                </div>
            </section>

            {/* Stats Section */}
            <section className="border-y border-border bg-card/20 px-6 py-12 backdrop-blur-sm">
                <div className="container mx-auto">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
                        {[
                            { label: 'Total Return', value: '+42.8%', color: 'text-green-400' },
                            { label: 'Sharpe Ratio', value: '2.45', color: 'text-text-primary' },
                            { label: 'Win Rate', value: '68%', color: 'text-text-primary' },
                            { label: 'Live Since', value: '2024', color: 'text-text-primary' },
                        ].map((stat, idx) => (
                            <div key={idx} className="text-center">
                                <div className={`text-3xl md:text-4xl font-bold font-mono mb-1 ${stat.color}`}>{stat.value}</div>
                                <div className="text-sm text-text-secondary uppercase tracking-wider">{stat.label}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* RL vs Classic ML Comparison Section */}
            <section className="py-24 bg-background relative overflow-hidden">
                <div className="container mx-auto px-6 max-w-6xl">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl md:text-5xl font-bold mb-6">Beyond Traditional Machine Learning</h2>
                        <p className="text-text-secondary max-w-2xl mx-auto text-lg">
                            Most algorithms try to predict the future. Our agent learns to <span className="text-accent font-semibold">shape its own destiny</span>.
                        </p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
                        {/* Traditional ML Card */}
                        <div className="p-8 rounded-2xl border border-dashed border-border bg-input/20 grayscale opacity-75 hover:grayscale-0 hover:opacity-100 transition-all duration-500">
                            <h3 className="text-2xl font-bold mb-6 text-text-secondary">Traditional ML</h3>
                            <ul className="space-y-4">
                                <li className="flex items-start gap-3">
                                    <span className="text-red-500 font-mono">✕</span>
                                    <div>
                                        <strong className="block text-text-primary">Optimizes for Accuracy</strong>
                                        <span className="text-sm text-text-secondary">Focuses on minimizing error, not maximizing profit.</span>
                                    </div>
                                </li>
                                <li className="flex items-start gap-3">
                                    <span className="text-red-500 font-mono">✕</span>
                                    <div>
                                        <strong className="block text-text-primary">Static Behavior</strong>
                                        <span className="text-sm text-text-secondary">Models degrade as market regimes change.</span>
                                    </div>
                                </li>
                                <li className="flex items-start gap-3">
                                    <span className="text-red-500 font-mono">✕</span>
                                    <div>
                                        <strong className="block text-text-primary">Single-Step Focus</strong>
                                        <span className="text-sm text-text-secondary">Makes decision based only on immediate next candle.</span>
                                    </div>
                                </li>
                            </ul>
                        </div>

                        {/* RL AlphaFlow Card */}
                        <div className="relative p-8 rounded-2xl bg-gradient-to-br from-accent/10 to-transparent border border-accent/30 shadow-2xl shadow-accent/5">
                            <div className="absolute -top-3 -right-3 bg-accent text-white text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wider shadow-lg">
                                Next Gen
                            </div>
                            <h3 className="text-2xl font-bold mb-6 text-white">Deep Reinforcement Learning</h3>
                            <ul className="space-y-4">
                                <li className="flex items-start gap-3">
                                    <span className="text-green-400 font-mono">✓</span>
                                    <div>
                                        <strong className="block text-white">Optimizes for Reward</strong>
                                        <span className="text-sm text-gray-300">Directly trained to maximize PnL and Sharpe Ratio.</span>
                                    </div>
                                </li>
                                <li className="flex items-start gap-3">
                                    <span className="text-green-400 font-mono">✓</span>
                                    <div>
                                        <strong className="block text-white">Dynamic Adaptation</strong>
                                        <span className="text-sm text-gray-300">Continually learns and adjusts to new volatility.</span>
                                    </div>
                                </li>
                                <li className="flex items-start gap-3">
                                    <span className="text-green-400 font-mono">✓</span>
                                    <div>
                                        <strong className="block text-white">Long-Horizon Planning</strong>
                                        <span className="text-sm text-gray-300">Considers future consequences of current trades.</span>
                                    </div>
                                </li>
                            </ul>
                        </div>
                    </div>

                    <ComparisonChart />
                </div>
            </section>

            {/* Feature Grid */}
            <section className="px-6 py-24 bg-background relative">
                <div className="container mx-auto max-w-6xl">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                        <FeatureCard
                            icon={TrendingUp}
                            title="Predictive Alpha"
                            description="Proprietary ensemble models analyze 18+ market indicators to forecast price movements with high confidence."
                        />
                        <FeatureCard
                            icon={Cpu}
                            title="Adaptive AI"
                            description="Our Reinforcement Learning agent learns from every trade, constantly refining its strategy for changing market conditions."
                        />
                        <FeatureCard
                            icon={Shield}
                            title="Risk Management"
                            description="Institutional-grade risk controls with dynamic stop-losses and draw-down protection built into the core logic."
                        />
                    </div>
                </div>
            </section>

            {/* CTA Section */}
            <section className="px-6 py-24">
                <div className="container mx-auto max-w-4xl">
                    <div className="bg-gradient-to-br from-card to-input border border-border rounded-2xl p-12 text-center relative overflow-hidden">
                        <div className="absolute top-0 right-0 w-64 h-64 bg-accent/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
                        <h2 className="text-3xl font-bold mb-6">Ready to upgrade your portfolio?</h2>
                        <p className="text-text-secondary mb-8 max-w-xl mx-auto">
                            Join exclusive investors leveraging our autonomous GBP/USD trading infrastructure.
                            Limited capacity available for the current fund.
                        </p>
                        <Link
                            to="/contact"
                            className="inline-flex items-center gap-2 px-8 py-3 bg-white text-black hover:bg-gray-200 font-bold rounded-lg transition-colors"
                        >
                            Start Now
                            <ChevronRight className="w-5 h-5" />
                        </Link>
                    </div>
                </div>
            </section>
        </div>
    );
}

function FeatureCard({ icon: Icon, title, description }: { icon: any, title: string, description: string }) {
    return (
        <motion.div
            whileHover={{ y: -5 }}
            className="bg-card border border-border rounded-xl p-8 hover:border-accent/50 transition-colors"
        >
            <div className="w-12 h-12 bg-accent/10 rounded-lg flex items-center justify-center mb-6">
                <Icon className="w-6 h-6 text-accent" />
            </div>
            <h3 className="text-xl font-bold mb-3">{title}</h3>
            <p className="text-text-secondary leading-relaxed">
                {description}
            </p>
        </motion.div>
    );
}
