import { useState, useEffect } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell, Line } from 'recharts';
import { Card } from './ui/Card';
import { Badge } from './ui/Badge';
import { Zap } from 'lucide-react';

// Types for our simulated data
interface MarketPoint {
    time: string;
    price: number;
    ema20: number;
    ema50: number;
    rsi: number;
    macd: number;
    signal: number;
    hist: number;
}

export function AnalysisPanel() {
    const [data, setData] = useState<MarketPoint[]>([]);
    const [strategies, setStrategies] = useState([
        { name: 'Momentum', active: true, type: 'positive' as const },
        { name: 'Trend Following', active: true, type: 'neutral' as const },
        { name: 'Mean Reversion', active: false, type: 'negative' as const },
    ]);

    const toggleStrategy = (name: string) => {
        setStrategies(prev => prev.map(s =>
            s.name === name ? { ...s, active: !s.active } : s
        ));
    };

    // Initial data generation
    useEffect(() => {
        const initialData: MarketPoint[] = [];
        let price = 1.2500;
        const now = new Date();

        for (let i = 60; i >= 0; i--) {
            const time = new Date(now.getTime() - i * 1000); // Past 60 seconds
            price = price + (Math.random() - 0.5) * 0.0005;
            initialData.push({
                time: time.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }),
                price: price,
                ema20: price - 0.0002,
                ema50: price - 0.0005,
                rsi: 40 + Math.random() * 20,
                macd: (Math.random() - 0.5) * 0.0002,
                signal: (Math.random() - 0.5) * 0.0002,
                hist: (Math.random() - 0.5) * 0.0001,
            });
        }
        setData(initialData);

        // Simulate live updates
        const interval = setInterval(() => {
            setData(prev => {
                const last = prev[prev.length - 1];
                const newPrice = last.price + (Math.random() - 0.5) * 0.0005;
                const newTime = new Date();

                const newPoint: MarketPoint = {
                    time: newTime.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }),
                    price: newPrice,
                    ema20: newPrice - 0.0002 + (Math.random() * 0.0001),
                    ema50: newPrice - 0.0005 + (Math.random() * 0.0001),
                    rsi: Math.max(0, Math.min(100, last.rsi + (Math.random() - 0.5) * 5)),
                    macd: last.macd + (Math.random() - 0.5) * 0.00005,
                    signal: last.signal + (Math.random() - 0.5) * 0.00005,
                    hist: 0 // Calc real hist if needed, but random is fine for visual demo
                };
                newPoint.hist = newPoint.macd - newPoint.signal;

                return [...prev.slice(1), newPoint];
            });
        }, 1000);

        return () => clearInterval(interval);
    }, []);

    if (data.length === 0) return <div>Loading analysis...</div>;

    const currentPrice = data[data.length - 1].price;
    const prevPrice = data[data.length - 2]?.price || currentPrice;
    const isUp = currentPrice >= prevPrice;

    return (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Main Price Chart */}
                <div className="lg:col-span-2">
                    <Card title="Live Market - GBP/USD" action={<Badge variant={isUp ? 'buy' : 'sell'} label={isUp ? 'BULLISH' : 'BEARISH'} />}>
                        <div className="h-[400px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={data}>
                                    <defs>
                                        <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor={isUp ? '#10b981' : '#ef4444'} stopOpacity={0.3} />
                                            <stop offset="95%" stopColor={isUp ? '#10b981' : '#ef4444'} stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#2a2a35" vertical={false} />
                                    <XAxis dataKey="time" stroke="#6b7280" tick={{ fontSize: 10 }} tickLine={false} axisLine={false} interval={10} />
                                    <YAxis domain={['auto', 'auto']} stroke="#6b7280" tick={{ fontSize: 10 }} tickLine={false} axisLine={false} width={60} tickFormatter={(val) => val.toFixed(4)} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#18181b', border: '1px solid #27272a' }}
                                        itemStyle={{ fontSize: '12px' }}
                                        labelStyle={{ color: '#a1a1aa', marginBottom: '4px' }}
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="price"
                                        stroke={isUp ? '#10b981' : '#ef4444'}
                                        strokeWidth={2}
                                        fillOpacity={1}
                                        fill="url(#colorPrice)"
                                        isAnimationActive={false}
                                    />
                                    <Line type="monotone" dataKey="ema20" stroke="#fbbf24" strokeWidth={1} dot={false} isAnimationActive={false} />
                                    <Line type="monotone" dataKey="ema50" stroke="#60a5fa" strokeWidth={1} dot={false} isAnimationActive={false} />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </Card>
                </div>

                {/* Side Metrics & Indicators */}
                <div className="flex flex-col gap-6">
                    <Card title="Market Pulse">
                        <div className="space-y-6">
                            <div>
                                <div className="text-sm text-text-secondary mb-1">Current Price</div>
                                <div className={`text-4xl font-mono font-bold ${isUp ? 'text-green-500' : 'text-red-500'}`}>
                                    {currentPrice.toFixed(5)}
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <IndicatorBox label="RSI (14)" value={data[data.length - 1].rsi.toFixed(1)} status={data[data.length - 1].rsi > 70 ? 'Overbought' : data[data.length - 1].rsi < 30 ? 'Oversold' : 'Neutral'} />
                                <IndicatorBox label="Volatility" value="Low" status="0.0012" />
                            </div>

                            <div>
                                <div className="text-xs font-semibold text-text-secondary uppercase mb-3 flex justify-between items-center">
                                    <span>Active Signals</span>
                                    <span className="text-[10px] bg-accent/10 text-accent px-1.5 py-0.5 rounded">INT</span>
                                </div>
                                <div className="space-y-2">
                                    {strategies.map(s => (
                                        <SignalRow
                                            key={s.name}
                                            name={s.name}
                                            active={s.active}
                                            type={s.type}
                                            onClick={() => toggleStrategy(s.name)}
                                        />
                                    ))}
                                </div>
                            </div>
                        </div>
                    </Card>

                    <Card title="MACD Momentum">
                        <div className="h-[140px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={data}>
                                    <Bar dataKey="hist" fill="#8884d8">
                                        {data.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.hist > 0 ? '#10b981' : '#ef4444'} />
                                        ))}
                                    </Bar>
                                    <Tooltip cursor={{ fill: 'transparent' }} contentStyle={{ backgroundColor: '#18181b', border: 'none' }} labelStyle={{ display: 'none' }} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </Card>
                </div>
            </div>
        </div>
    );
}

function IndicatorBox({ label, value, status }: { label: string, value: string, status: string }) {
    return (
        <div className="bg-input/50 p-3 rounded-lg border border-border">
            <div className="text-xs text-text-secondary mb-1">{label}</div>
            <div className="font-bold text-lg">{value}</div>
            <div className="text-[10px] text-accent mt-1">{status}</div>
        </div>
    );
}

function SignalRow({ name, active, type, onClick }: { name: string, active: boolean, type: 'positive' | 'negative' | 'neutral', onClick: () => void }) {
    const color = type === 'positive' ? 'bg-green-500' : type === 'negative' ? 'bg-red-500' : 'bg-gray-500';
    return (
        <div
            onClick={onClick}
            className={`flex items-center justify-between p-2 rounded bg-input/30 cursor-pointer hover:bg-input/50 transition-all border ${active ? 'border-accent/30' : 'border-transparent opacity-60 hover:opacity-100'}`}
        >
            <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${active ? color : 'bg-gray-700'}`} />
                <span className={`text-sm ${active ? 'text-text-primary' : 'text-text-secondary'}`}>{name}</span>
            </div>
            {active && <Zap className="w-3 h-3 text-yellow-500 animate-pulse" />}
        </div>
    );
}
