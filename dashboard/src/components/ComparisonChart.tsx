import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

const data = [
    { pnl_rl: 0, pnl_ml: 0, market: 0, time: 'Day 1' },
    { pnl_rl: 12, pnl_ml: 10, market: 2, time: 'Day 5' },
    { pnl_rl: 25, pnl_ml: 22, market: 5, time: 'Day 10' },
    { pnl_rl: 45, pnl_ml: 40, market: 12, time: 'Day 20' },
    // Market Crash Scenario
    { pnl_rl: 52, pnl_ml: 28, market: -15, time: 'Volatility Spike' },
    { pnl_rl: 68, pnl_ml: 15, market: -25, time: 'Day 30' },
    { pnl_rl: 85, pnl_ml: 5, market: -10, time: 'Day 40' },
    { pnl_rl: 110, pnl_ml: -12, market: 5, time: 'Recovery' },
];

export function ComparisonChart() {
    return (
        <div className="w-full h-[400px] mt-12 bg-card/50 border border-border rounded-2xl p-6">
            <div className="flex items-center justify-between mb-8">
                <div>
                    <h3 className="text-xl font-bold">Performance Stress Test</h3>
                    <p className="text-sm text-text-secondary">Simulated reaction to -25% market drawdown</p>
                </div>
                <div className="flex gap-4 text-sm">
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-accent" />
                        <span>AlphaFlow RL</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-gray-500" />
                        <span>Classic ML</span>
                    </div>
                </div>
            </div>

            <ResponsiveContainer width="100%" height="85%">
                <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                    <defs>
                        <linearGradient id="colorRL" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="colorML" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#6b7280" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#6b7280" stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2a2a35" vertical={false} />
                    <XAxis dataKey="time" stroke="#6b7280" tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
                    <YAxis stroke="#6b7280" tick={{ fontSize: 12 }} tickLine={false} axisLine={false} tickFormatter={(val) => `${val}%`} />
                    <Tooltip
                        contentStyle={{ backgroundColor: '#18181b', border: '1px solid #27272a' }}
                        formatter={(value: any) => [`${value}%`, 'Return']}
                    />
                    <Area
                        type="monotone"
                        dataKey="pnl_rl"
                        name="AlphaFlow RL"
                        stroke="#3b82f6"
                        strokeWidth={3}
                        fillOpacity={1}
                        fill="url(#colorRL)"
                    />
                    <Area
                        type="monotone"
                        dataKey="pnl_ml"
                        name="Classic ML"
                        stroke="#6b7280"
                        strokeWidth={2}
                        fillOpacity={1}
                        fill="url(#colorML)"
                        strokeDasharray="5 5"
                    />
                </AreaChart>
            </ResponsiveContainer>
        </div>
    );
}
