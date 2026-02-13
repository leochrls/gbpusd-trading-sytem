import { Card } from './ui/Card';
import type { PredictionResponse } from '../api';
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
    PieChart, Pie, Legend,
    LineChart, Line, ReferenceLine
} from 'recharts';

interface AnalyticsChartsProps {
    history: PredictionResponse[];
}

export function AnalyticsCharts({ history }: AnalyticsChartsProps) {
    const signalColors = {
        BUY: '#00d4aa',
        SELL: '#ff4d6d',
        HOLD: '#8892a4'
    };

    // Prepare data for Signal History (Bar Chart)
    const historyData = history.map((item, index) => ({
        id: index + 1,
        signal: item.action,
        confidence: item.confidence,
        time: new Date(item.timestamp).toLocaleTimeString(),
    }));

    // Prepare data for Signal Distribution (Pie Chart)
    const distributionData = [
        { name: 'BUY', value: history.filter(h => h.action === 'BUY').length },
        { name: 'SELL', value: history.filter(h => h.action === 'SELL').length },
        { name: 'HOLD', value: history.filter(h => h.action === 'HOLD').length },
    ].filter(d => d.value > 0);

    // Prepare data for Confidence Trend (Line Chart)
    const confidenceData = history.map((item, index) => ({
        time: index + 1, // Simple index for X-axis
        confidence: item.confidence * 100,
        signal: item.action
    }));

    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            {/* 1. Signal History */}
            <Card title="Signal History">
                <div className="h-[200px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={historyData}>
                            <XAxis dataKey="id" stroke="#8892a4" fontSize={10} tickLine={false} axisLine={false} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#12121a', border: '1px solid #1e1e2e', borderRadius: '8px' }}
                                itemStyle={{ color: '#e2e8f0' }}
                                cursor={{ fill: 'rgba(255, 255, 255, 0.05)' }}
                            />
                            <Bar dataKey="confidence" radius={[4, 4, 0, 0]}>
                                {historyData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={signalColors[entry.signal as keyof typeof signalColors] || '#8892a4'} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </Card>

            {/* 2. Signal Distribution */}
            <Card title="Distribution">
                <div className="h-[200px] w-full relative">
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                data={distributionData}
                                cx="50%"
                                cy="50%"
                                innerRadius={60}
                                outerRadius={80}
                                paddingAngle={5}
                                dataKey="value"
                                stroke="none"
                            >
                                {distributionData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={signalColors[entry.name as keyof typeof signalColors]} />
                                ))}
                            </Pie>
                            <Tooltip
                                contentStyle={{ backgroundColor: '#12121a', border: '1px solid #1e1e2e', borderRadius: '8px' }}
                                itemStyle={{ color: '#e2e8f0' }}
                            />
                            <Legend verticalAlign="bottom" iconSize={8} wrapperStyle={{ fontSize: '10px' }} />
                        </PieChart>
                    </ResponsiveContainer>
                    {/* Center Text */}
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                        <span className="text-2xl font-bold font-mono text-text-primary">{history.length}</span>
                    </div>
                </div>
            </Card>

            {/* 3. Confidence Trend */}
            <Card title="Confidence Trend">
                <div className="h-[200px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={confidenceData}>
                            <XAxis dataKey="time" stroke="#8892a4" fontSize={10} tickLine={false} axisLine={false} />
                            <YAxis hide domain={[0, 100]} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#12121a', border: '1px solid #1e1e2e', borderRadius: '8px' }}
                                itemStyle={{ color: '#e2e8f0' }}
                            />
                            <ReferenceLine y={50} stroke="#8892a4" strokeDasharray="3 3" />
                            <Line
                                type="monotone"
                                dataKey="confidence"
                                stroke="#3b82f6"
                                strokeWidth={2}
                                dot={{ r: 3, fill: '#3b82f6', strokeWidth: 0 }}
                                activeDot={{ r: 5, strokeWidth: 0 }}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </Card>
        </div>
    );
}
