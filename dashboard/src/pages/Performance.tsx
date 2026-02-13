import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';


// Mock performance data
const data = [
    { month: 'Jan', ret: 2.3, equity: 102.3 },
    { month: 'Feb', ret: 4.1, equity: 106.5 },
    { month: 'Mar', ret: -1.2, equity: 105.2 },
    { month: 'Apr', ret: 5.4, equity: 110.9 },
    { month: 'May', ret: 3.2, equity: 114.4 },
    { month: 'Jun', ret: 0.8, equity: 115.3 },
    { month: 'Jul', ret: 6.1, equity: 122.3 },
    { month: 'Aug', ret: -0.5, equity: 121.7 },
    { month: 'Sep', ret: 4.8, equity: 127.5 },
    { month: 'Oct', ret: 3.5, equity: 132.0 },
    { month: 'Nov', ret: 2.1, equity: 134.8 },
    { month: 'Dec', ret: 5.2, equity: 141.8 },
];

export function Performance() {
    return (
        <div className="container mx-auto px-6 py-12">
            <div className="text-center mb-16">
                <h1 className="text-4xl font-bold mb-4">Transparent Performance</h1>
                <p className="text-text-secondary max-w-2xl mx-auto">
                    Track our audited trading results. We believe in complete transparency,
                    providing real-time access to our fund's performance metrics.
                </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-16">
                {/* Main Chart */}
                <div className="lg:col-span-2 bg-card border border-border rounded-xl p-6 shadow-lg">
                    <h3 className="text-lg font-semibold mb-6">Cumulative Return (2025)</h3>
                    <div className="h-[400px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={data}>
                                <defs>
                                    <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <XAxis dataKey="month" stroke="#8892a4" fontSize={12} tickLine={false} axisLine={false} />
                                <YAxis stroke="#8892a4" fontSize={12} tickLine={false} axisLine={false} domain={['auto', 'auto']} />
                                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" vertical={false} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#12121a', border: '1px solid #1e1e2e', borderRadius: '8px' }}
                                />
                                <Area
                                    type="monotone"
                                    dataKey="equity"
                                    stroke="#3b82f6"
                                    strokeWidth={3}
                                    fillOpacity={1}
                                    fill="url(#colorEquity)"
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Monthly Returns Table */}
                <div className="bg-card border border-border rounded-xl p-6 shadow-lg">
                    <h3 className="text-lg font-semibold mb-6">Monthly Breakdown</h3>
                    <div className="overflow-y-auto h-[400px] custom-scrollbar">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="text-text-secondary border-b border-border">
                                    <th className="text-left py-2 font-medium">Month</th>
                                    <th className="text-right py-2 font-medium">Return</th>
                                    <th className="text-right py-2 font-medium">NAV</th>
                                </tr>
                            </thead>
                            <tbody>
                                {data.map((item) => (
                                    <tr key={item.month} className="border-b border-border/50 hover:bg-input/20 transition-colors">
                                        <td className="py-3">{item.month} 2025</td>
                                        <td className={`text-right py-3 font-mono font-medium ${item.ret >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                            {item.ret > 0 ? '+' : ''}{item.ret}%
                                        </td>
                                        <td className="text-right py-3 font-mono text-text-secondary">
                                            {item.equity.toFixed(2)}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                <MetricCard label="YTD Return" value="+41.8%" subtext="Year to Date" />
                <MetricCard label="Max Drawdown" value="-4.2%" subtext="Peak to Trough" />
                <MetricCard label="Sharpe Ratio" value="2.85" subtext="Risk Adjusted" />
                <MetricCard label="Sortino Ratio" value="3.12" subtext="Downside Risk" />
            </div>
        </div>
    );
}

function MetricCard({ label, value, subtext }: { label: string, value: string, subtext: string }) {
    return (
        <div className="bg-card border border-border p-6 rounded-xl text-center">
            <div className="text-sm text-text-secondary uppercase tracking-wider mb-2">{label}</div>
            <div className="text-3xl font-bold text-text-primary mb-1">{value}</div>
            <div className="text-xs text-text-secondary">{subtext}</div>
        </div>
    );
}
