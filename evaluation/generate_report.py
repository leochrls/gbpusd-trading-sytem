"""
Genere le rapport HTML final complet et autonome.
"""
import json
from datetime import datetime
from pathlib import Path

from loguru import logger


def generate_final_report() -> None:
    """Genere un rapport HTML autonome avec toutes les analyses."""

    # Chargement metriques
    with open('evaluation/final_metrics.json', 'r') as f:
        metrics_raw = json.load(f)

    # Chargement resultats ML si dispo
    ml_results = {}
    ml_path = Path('evaluation/ml_results.json')
    if ml_path.exists():
        with open(ml_path, 'r') as f:
            ml_results = json.load(f)

    # Chargement resultats RL si dispo
    rl_results = {}
    rl_path = Path('evaluation/rl_test_results.json')
    if rl_path.exists():
        with open(rl_path, 'r') as f:
            rl_results = json.load(f)

    # Chargement EDA stats si dispo
    eda_stats = {}
    eda_path = Path('data/processed/summary_stats.json')
    if eda_path.exists():
        with open(eda_path, 'r') as f:
            eda_stats = json.load(f)

    # Build tableau HTML
    rows_html = ""
    best_sharpe = -999.0
    best_name = ""

    for name, metrics in metrics_raw.items():
        sharpe = metrics.get('sharpe_ratio', 0)
        ret = metrics.get('total_return_pct', 0)
        dd = metrics.get('max_drawdown_pct', 0)
        pf = metrics.get('profit_factor', 0)
        wr = metrics.get('win_rate_pct', 0)
        trades = metrics.get('n_trades', 0)
        calmar = metrics.get('calmar_ratio', 0)

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_name = name

        ret_color = '#00ff88' if ret >= 0 else '#ff4444'
        dd_color = '#ff4444' if dd < -10 else '#ffaa00' if dd < -5 else '#00ff88'

        rows_html += f"""
        <tr>
            <td><strong>{name}</strong></td>
            <td style="color:{ret_color}">{ret:+.2f}%</td>
            <td style="color:{dd_color}">{dd:.2f}%</td>
            <td>{sharpe:.4f}</td>
            <td>{calmar:.4f}</td>
            <td>{pf:.4f}</td>
            <td>{wr:.1f}%</td>
            <td>{trades}</td>
        </tr>
        """

    # ML best model info
    ml_section = ""
    if ml_results:
        best_ml = ml_results.get('best_model', 'N/A')
        best_auc = ml_results.get('best_val_roc_auc', 0)
        ml_section = f"""
        <div class="stat-box">
            <h2>Machine Learning</h2>
            <p>Meilleur modele : <strong>{best_ml}</strong>
               (Val ROC-AUC : {best_auc:.4f})</p>
            <p style="color:#8b949e">
                Les modeles ML souffrent de sur-trading (~8000+ trades)
                et d'overfitting (train ROC-AUC ~0.82, val ~0.52).
            </p>
        </div>
        """

    # RL section
    rl_section = ""
    if rl_results:
        rl_ret = rl_results.get('total_return_pct', 0)
        rl_sharpe = rl_results.get('sharpe_ratio', 0)
        rl_trades = rl_results.get('n_trades', 0)
        rl_section = f"""
        <div class="stat-box">
            <h2>Reinforcement Learning (DQN)</h2>
            <p>Return test 2024 : <strong>{rl_ret:+.2f}%</strong> |
               Sharpe : <strong>{rl_sharpe:.4f}</strong> |
               Trades : <strong>{rl_trades}</strong></p>
            <p style="color:#8b949e">
                L'agent DQN a appris a trader moins frequemment,
                limitant les couts de transaction.
            </p>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Rapport Final - GBP/USD Trading System</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            padding: 40px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{ color: #58a6ff; font-size: 2em; margin-bottom: 10px; }}
        h2 {{ color: #79c0ff; margin: 20px 0 10px; }}
        .header {{
            border-bottom: 2px solid #21262d;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            margin: 5px;
        }}
        .badge-green {{ background: #1a4a2e; color: #00ff88; }}
        .badge-blue {{ background: #1a2d4a; color: #4488ff; }}
        .stat-box {{
            background: #161b22;
            border: 1px solid #21262d;
            border-radius: 8px;
            padding: 24px;
            margin: 20px 0;
        }}
        .highlight-box {{
            background: #1a3a1a;
            border: 2px solid #00ff88;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 0.95em;
        }}
        th {{
            background: #21262d;
            color: #79c0ff;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #30363d;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #21262d;
        }}
        tr:hover {{ background: #1c2128; }}
        img {{
            max-width: 100%;
            border-radius: 8px;
            margin: 15px 0;
            border: 1px solid #21262d;
        }}
        .metric-card {{
            background: #161b22;
            border: 1px solid #21262d;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #58a6ff;
        }}
        .metric-label {{
            color: #8b949e;
            font-size: 0.85em;
            margin-top: 5px;
        }}
        footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #21262d;
            color: #8b949e;
            font-size: 0.85em;
        }}
    </style>
</head>
<body>

    <div class="header">
        <h1>Rapport Final - GBP/USD Trading System</h1>
        <p style="color:#8b949e">Genere le {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
        <br>
        <span class="badge badge-green">2022 - Train</span>
        <span class="badge badge-blue">2023 - Validation</span>
        <span class="badge" style="background:#4a1a1a;color:#ff4444">2024 - Test Final</span>
    </div>

    <!-- RESUME EXECUTIF -->
    <div class="highlight-box">
        <h2>Meilleure Strategie : {best_name}</h2>
        <p>Sharpe Ratio sur Test 2024 : <strong>{best_sharpe:.4f}</strong></p>
        <p style="color:#8b949e;margin-top:8px">
            Un modele valide doit etre robuste sur 2024, tenir compte des couts
            de transaction, et eviter l'overfitting temporel.
        </p>
    </div>

    <!-- TABLEAU COMPARATIF -->
    <div class="stat-box">
        <h2>Comparaison Finale - Test 2024</h2>
        <table>
            <thead>
                <tr>
                    <th>Strategie</th>
                    <th>Return Total</th>
                    <th>Max Drawdown</th>
                    <th>Sharpe Ratio</th>
                    <th>Calmar Ratio</th>
                    <th>Profit Factor</th>
                    <th>Win Rate</th>
                    <th>Nb Trades</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>

    <!-- ML -->
    {ml_section}

    <!-- RL -->
    {rl_section}

    <!-- CONCLUSION -->
    <div class="stat-box">
        <h2>Conclusions</h2>
        <ul style="line-height:1.8">
            <li>Les modeles ML (ROC-AUC ~0.52) ne depassent pas significativement
                le hasard sur M15 GBP/USD.</li>
            <li>Le sur-trading des modeles ML detruit la performance via les
                couts de transaction.</li>
            <li>L'agent RL (DQN) apprend a limiter le nombre de trades,
                reduisant l'impact des couts.</li>
            <li>Buy &amp; Hold reste un benchmark difficile a battre sur
                les paires de devises majeures.</li>
            <li>La strategie RuleBased (EMA crossover + RSI) offre un
                compromis raisonnable.</li>
        </ul>
    </div>

    <footer>
        <p>GBP/USD Trading System - Projet Fil Rouge M1 - Fevrier 2026</p>
        <p>Split temporel strict : Train 2022 | Val 2023 | Test 2024</p>
        <p>Transaction cost : 0.02% (2 pips) | Slippage : 0.01%</p>
    </footer>

</body>
</html>"""

    output_path = 'evaluation/final_report.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    logger.success(f"Rapport final genere : {output_path}")
