import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

interface BadgeProps {
    label: string;
    variant?: 'ml' | 'rl' | 'buy' | 'sell' | 'hold' | 'default' | 'success' | 'danger' | 'warning';
    className?: string;
}

export function Badge({ label, variant = 'default', className }: BadgeProps) {
    const styles = {
        ml: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
        rl: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
        buy: 'bg-[#00d4aa]/10 text-[#00d4aa] border-[#00d4aa]/20',
        sell: 'bg-[#ff4d6d]/10 text-[#ff4d6d] border-[#ff4d6d]/20',
        hold: 'bg-[#8892a4]/10 text-[#8892a4] border-[#8892a4]/20',
        success: 'bg-green-500/10 text-green-400 border-green-500/20',
        danger: 'bg-red-500/10 text-red-400 border-red-500/20',
        warning: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
        default: 'bg-gray-700/30 text-gray-300 border-gray-600/30',
    };

    return (
        <span className={cn("px-2.5 py-0.5 rounded-full text-xs font-semibold border uppercase tracking-wide", styles[variant], className)}>
            {label}
        </span>
    );
}
