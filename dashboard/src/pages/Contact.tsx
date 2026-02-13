import { Check, Mail, MessageSquare } from 'lucide-react';

export function Contact() {
    return (
        <div className="container mx-auto px-6 py-12">
            <div className="text-center mb-16">
                <h1 className="text-4xl font-bold mb-4">Invest with AlphaFlow</h1>
                <p className="text-text-secondary max-w-xl mx-auto">
                    Choose a plan that fits your capital allocation. Institutional inquiries please contact sales directly.
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto mb-20">
                <PricingCard
                    title="Starter"
                    price="2%"
                    sub="Mgmt Fee"
                    description="For individual investors starting their journey."
                    features={['Min. Investment $10k', 'Monthly Liquidity', 'Standard Reporting', 'Email Support']}
                />
                <PricingCard
                    title="Professional"
                    price="1.5%"
                    sub="Mgmt Fee + 20% Perf"
                    description="For serious investors seeking maximum returns."
                    features={['Min. Investment $100k', 'Weekly Liquidity', 'Real-time Dashboard', 'Priority Support']}
                    popular
                />
                <PricingCard
                    title="Institutional"
                    price="Custom"
                    sub="Fee Structure"
                    description="Tailored solutions for funds and family offices."
                    features={['Min. Investment $1M+', 'Daily Liquidity', 'Custom API Access', 'Dedicated Account Mgr']}
                />
            </div>

            <div className="bg-card border border-border rounded-2xl p-12 text-center max-w-3xl mx-auto">
                <h2 className="text-2xl font-bold mb-8">Get in Touch</h2>
                <form className="max-w-md mx-auto space-y-4 text-left">
                    <div>
                        <label className="block text-sm font-medium text-text-secondary mb-1">Email Address</label>
                        <div className="relative">
                            <input type="email" className="w-full bg-input border border-border rounded-lg px-4 py-2 pl-10 focus:outline-none focus:border-accent transition-colors" placeholder="you@company.com" />
                            <Mail className="w-4 h-4 text-text-secondary absolute left-3 top-3" />
                        </div>
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-text-secondary mb-1">Message</label>
                        <div className="relative">
                            <textarea className="w-full bg-input border border-border rounded-lg px-4 py-2 pl-10 h-32 focus:outline-none focus:border-accent transition-colors" placeholder="Tell us about your investment goals..." />
                            <MessageSquare className="w-4 h-4 text-text-secondary absolute left-3 top-3" />
                        </div>
                    </div>
                    <button className="w-full bg-accent hover:bg-blue-600 text-white font-bold py-3 rounded-lg transition-colors">
                        Send Message
                    </button>
                </form>
            </div>
        </div>
    );
}

function PricingCard({ title, price, sub, description, features, popular }: { title: string, price: string, sub: string, description: string, features: string[], popular?: boolean }) {
    return (
        <div className={`bg-card border rounded-xl p-8 relative flex flex-col ${popular ? 'border-accent shadow-lg shadow-blue-500/10' : 'border-border'}`}>
            {popular && (
                <div className="absolute -top-4 left-1/2 -translate-x-1/2 bg-accent text-white text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wider">
                    Most Popular
                </div>
            )}
            <h3 className="text-xl font-bold mb-2">{title}</h3>
            <p className="text-text-secondary text-sm mb-6 h-10">{description}</p>
            <div className="mb-6">
                <span className="text-4xl font-bold">{price}</span>
                <span className="text-text-secondary ml-2 text-sm">{sub}</span>
            </div>
            <ul className="space-y-4 mb-8 flex-1">
                {features.map((feat, i) => (
                    <li key={i} className="flex items-center gap-3 text-sm">
                        <Check className="w-4 h-4 text-green-400" />
                        {feat}
                    </li>
                ))}
            </ul>
            <button className={`w-full py-3 rounded-lg font-bold transition-all ${popular ? 'bg-accent hover:bg-blue-600 text-white' : 'bg-input hover:bg-border text-text-primary'}`}>
                Choose Plan
            </button>
        </div>
    );
}
