import { Link, Outlet, useLocation } from 'react-router-dom';
import { Activity, BarChart2, BookOpen, Mail, Menu, X } from 'lucide-react';
import { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';

export function PublicLayout() {
    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const location = useLocation();

    const navLinks = [
        { name: 'Home', path: '/', icon: Activity },
        { name: 'Performance', path: '/performance', icon: BarChart2 },
        { name: 'Strategy', path: '/strategy', icon: BookOpen },
        { name: 'Contact', path: '/contact', icon: Mail },
    ];

    return (
        <div className="min-h-screen bg-background text-text-primary font-sans flex flex-col">
            {/* Navigation */}
            <nav className="fixed top-0 w-full z-50 bg-background/80 backdrop-blur-md border-b border-border">
                <div className="container mx-auto px-6 h-16 flex items-center justify-between">
                    <Link to="/" className="flex items-center gap-2 font-bold text-xl tracking-tight">
                        <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center">
                            <Activity className="w-5 h-5 text-white" />
                        </div>
                        <span>AlphaFlow<span className="text-accent">.ai</span></span>
                    </Link>

                    {/* Desktop Nav */}
                    <div className="hidden md:flex items-center gap-8">
                        {navLinks.map(link => (
                            <Link
                                key={link.path}
                                to={link.path}
                                className={`text-sm font-medium transition-colors hover:text-accent ${location.pathname === link.path ? 'text-accent' : 'text-text-secondary'
                                    }`}
                            >
                                {link.name}
                            </Link>
                        ))}
                        <Link
                            to="/dashboard"
                            className="px-4 py-2 bg-input border border-border rounded-lg text-sm font-medium hover:bg-accent hover:text-white transition-all ml-4"
                        >
                            Client Login
                        </Link>
                    </div>

                    {/* Mobile Menu Button */}
                    <button
                        className="md:hidden p-2 text-text-secondary"
                        onClick={() => setIsMenuOpen(!isMenuOpen)}
                    >
                        {isMenuOpen ? <X /> : <Menu />}
                    </button>
                </div>

                {/* Mobile Nav */}
                <AnimatePresence>
                    {isMenuOpen && (
                        <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            className="md:hidden border-b border-border bg-card overflow-hidden"
                        >
                            <div className="flex flex-col p-4 gap-4">
                                {navLinks.map(link => (
                                    <Link
                                        key={link.path}
                                        to={link.path}
                                        onClick={() => setIsMenuOpen(false)}
                                        className={`flex items-center gap-3 px-4 py-3 rounded-lg ${location.pathname === link.path ? 'bg-accent/10 text-accent' : 'text-text-secondary hover:bg-input'
                                            }`}
                                    >
                                        <link.icon className="w-5 h-5" />
                                        {link.name}
                                    </Link>
                                ))}
                                <Link
                                    to="/dashboard"
                                    onClick={() => setIsMenuOpen(false)}
                                    className="mt-2 w-full py-3 bg-accent text-white font-medium rounded-lg text-center"
                                >
                                    Client Portal
                                </Link>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </nav>

            {/* Main Content */}
            <main className="flex-1 pt-16">
                <Outlet />
            </main>

            {/* Footer */}
            <footer className="border-t border-border bg-card/30">
                <div className="container mx-auto px-6 py-12">
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
                        <div className="col-span-1 md:col-span-2">
                            <div className="flex items-center gap-2 font-bold text-lg mb-4">
                                <div className="w-6 h-6 rounded bg-accent/80 flex items-center justify-center">
                                    <Activity className="w-4 h-4 text-white" />
                                </div>
                                <span>AlphaFlow.ai</span>
                            </div>
                            <p className="text-text-secondary text-sm max-w-sm">
                                Next-generation algorithmic trading powered by Deep Reinforcement Learning.
                                Adaptive strategies for volatile markets.
                            </p>
                        </div>
                        <div>
                            <h4 className="font-semibold mb-4">Platform</h4>
                            <ul className="space-y-2 text-sm text-text-secondary">
                                <li><Link to="/performance" className="hover:text-accent">Performance</Link></li>
                                <li><Link to="/strategy" className="hover:text-accent">Methodology</Link></li>
                                <li><Link to="/dashboard" className="hover:text-accent">Live Dashboard</Link></li>
                            </ul>
                        </div>
                        <div>
                            <h4 className="font-semibold mb-4">Legal</h4>
                            <ul className="space-y-2 text-sm text-text-secondary">
                                <li><a href="#" className="hover:text-accent">Terms of Service</a></li>
                                <li><a href="#" className="hover:text-accent">Privacy Policy</a></li>
                                <li><a href="#" className="hover:text-accent">Risk Disclosure</a></li>
                            </ul>
                        </div>
                    </div>
                    <div className="pt-8 border-t border-border flex flex-col md:flex-row justify-between items-center gap-4 text-xs text-text-secondary">
                        <p>© 2026 AlphaFlow Technologies. All rights reserved.</p>
                        <p>Trading Forex involves substantial risk of loss.</p>
                    </div>
                </div>
            </footer>
        </div>
    );
}
