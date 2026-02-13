import { RefreshCw, Server, Clock } from 'lucide-react';
import { Badge } from './ui/Badge';

interface HeaderProps {
    isConnected: boolean;
    uptime: string;
    totalRequests: number;
    activeModel?: { name: string; type: string };
    onRefresh: () => void;
    isRefreshing: boolean;
}

export function Header({ isConnected, uptime, totalRequests, activeModel, onRefresh, isRefreshing }: HeaderProps) {
    return (
        <header className="sticky top-0 z-50 w-full bg-background/80 backdrop-blur-md border-b border-border px-6 py-3 flex items-center justify-between">
            <div className="flex items-center gap-4">
                <h1 className="text-xl font-bold tracking-tight text-text-primary">GBP/USD Trading Dashboard</h1>

                <div className="flex items-center gap-2 pl-4 border-l border-border/50">
                    <div className={`w-2.5 h-2.5 rounded-full ${isConnected ? 'bg-signal-buy shadow-[0_0_8px_rgba(0,212,170,0.5)]' : 'bg-signal-sell'} animate-pulse`} />
                    <span className={`text-sm font-medium ${isConnected ? 'text-signal-buy' : 'text-signal-sell'}`}>
                        {isConnected ? 'Online' : 'Offline'}
                    </span>
                </div>
            </div>

            <div className="flex items-center gap-6">
                {activeModel && (
                    <div className="flex items-center gap-2">
                        <span className="text-sm text-text-secondary">Active Model:</span>
                        <span className="text-sm font-medium text-text-primary">{activeModel.name}</span>
                        <Badge label={activeModel.type} variant={activeModel.type.toLowerCase() === 'rl' ? 'rl' : 'ml'} />
                    </div>
                )}

                <div className="flex items-center gap-4 text-sm text-text-secondary bg-card/50 px-3 py-1.5 rounded-lg border border-border/50">
                    <div className="flex items-center gap-2">
                        <Clock className="w-4 h-4 text-accent" />
                        <span className="font-mono text-text-primary">{uptime}</span>
                    </div>
                    <div className="w-px h-4 bg-border/50" />
                    <div className="flex items-center gap-2">
                        <Server className="w-4 h-4 text-accent" />
                        <span className="font-mono text-text-primary">{totalRequests}</span>
                        <span className="text-xs">reqs</span>
                    </div>
                </div>

                <button
                    onClick={onRefresh}
                    className={`p-2 rounded-lg hover:bg-white/5 text-text-secondary hover:text-text-primary transition-colors ${isRefreshing ? 'animate-spin' : ''}`}
                    title="Refresh Data"
                >
                    <RefreshCw className="w-5 h-5" />
                </button>
            </div>
        </header>
    );
}
