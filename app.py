from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
from tp2 import mm1_metrics, calculate_probabilities, mm_s_metrics, calculate_probabilities_mm_s
from tp3 import simulate as simulate_mm1

app = Flask(__name__)

COLORS = ['#3b82f6', '#06b6d4', '#a78bfa', '#34d399', '#f97316', '#f43f5e']

def simulate_markov(P, pi0, steps):
    P = np.array(P, dtype=float)
    pi = np.array(pi0, dtype=float)
    n = len(pi)
    history = [pi.copy()]
    for _ in range(steps):
        pi = np.dot(pi, P)
        history.append(pi.copy())
    return history

def compute_steady_state(P):
    P = np.array(P, dtype=float)
    n = P.shape[0]
    # Solve πP = π, sum(π)=1
    A = (P.T - np.eye(n))
    A = np.vstack([A, np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1
    try:
        steady, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        steady = np.abs(steady)
        steady /= steady.sum()
        return steady.tolist()
    except:
        return [1/n] * n

def generate_chart(history, steps, state_names=None):
    n = len(history[0])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((13/255, 21/255, 41/255, 0.78))

    x = list(range(steps + 1))
    if not state_names or len(state_names) != n:
        state_names = [f'State {i+1}' for i in range(n)]

    for i in range(n):
        y = [h[i] for h in history]
        color = COLORS[i % len(COLORS)]
        ax.plot(x, y, color=color, linewidth=2.2, label=state_names[i],
                marker='o', markersize=3.5, markerfacecolor=color,
                markeredgewidth=0, alpha=0.92)
        ax.fill_between(x, y, alpha=0.08, color=color)

    ax.set_xlabel('Steps', color='#8ba3cc', fontsize=10, labelpad=8)
    ax.set_ylabel('Probability', color='#8ba3cc', fontsize=10, labelpad=8)
    ax.tick_params(colors='#8ba3cc', labelsize=9)
    ax.spines['bottom'].set_color('#1e3a5f')
    ax.spines['left'].set_color('#1e3a5f')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0, steps)
    ax.grid(axis='y', color='#1e3a5f', linewidth=0.6, alpha=0.5)
    ax.grid(axis='x', color='#1e3a5f', linewidth=0.4, alpha=0.3)

    # Place legend below the chart so it never overlaps the lines
    legend = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(n, 4),
        framealpha=0.18,
        edgecolor='#1e3a5f',
        labelcolor='#c9deff',
        fontsize=9,
        fancybox=True
    )
    legend.get_frame().set_facecolor('#0d1f3c')

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
    fig.tight_layout(pad=1.5)


    buf = io.BytesIO()
    plt.savefig(
        buf,
        format='png',
        dpi=115,
        bbox_inches='tight',
        transparent=True
    )
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64


def generate_probability_bar_chart(probabilities):
    x = list(range(len(probabilities)))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((13 / 255, 21 / 255, 41 / 255, 0.78))

    bars = ax.bar(x, probabilities, color='#06b6d4', edgecolor='#67e8f9', linewidth=1.0, alpha=0.9)
    for bar in bars:
        bar.set_linewidth(0.8)

    ax.set_xlabel('n', color='#8ba3cc', fontsize=10, labelpad=8)
    ax.set_ylabel('P(N = n)', color='#8ba3cc', fontsize=10, labelpad=8)
    ax.tick_params(colors='#8ba3cc', labelsize=9)
    ax.spines['bottom'].set_color('#1e3a5f')
    ax.spines['left'].set_color('#1e3a5f')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#1e3a5f', linewidth=0.6, alpha=0.5)
    ax.grid(axis='x', color='#1e3a5f', linewidth=0.2, alpha=0.15)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=12))

    fig.tight_layout(pad=1.4)
    buf = io.BytesIO()
    plt.savefig(
        buf,
        format='png',
        dpi=115,
        bbox_inches='tight',
        transparent=True
    )
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

def generate_tp3_metrics_chart(metrics):
    labels = ['N', 'Nq', 'T', 'Tq', 'Ts', 'SU']
    values = [float(metrics.get(k, 0.0)) for k in labels]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((13 / 255, 21 / 255, 41 / 255, 0.78))

    bar_colors = ['#3b82f6', '#06b6d4', '#0ea5e9', '#14b8a6', '#22d3a0', '#f59e0b']
    bars = ax.bar(labels, values, color=bar_colors, edgecolor='#c9deff', linewidth=0.6, alpha=0.92)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{value:.3f}',
            ha='center',
            va='bottom',
            color='#e8f0ff',
            fontsize=9
        )

    ax.set_ylabel('Value', color='#8ba3cc', fontsize=10, labelpad=8)
    ax.tick_params(colors='#8ba3cc', labelsize=10)
    ax.spines['bottom'].set_color('#1e3a5f')
    ax.spines['left'].set_color('#1e3a5f')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#1e3a5f', linewidth=0.6, alpha=0.5)
    ax.set_ylim(0, max(values + [1.0]) * 1.2)

    fig.tight_layout(pad=1.4)
    buf = io.BytesIO()
    plt.savefig(
        buf,
        format='png',
        dpi=115,
        bbox_inches='tight',
        transparent=True
    )
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        data = request.get_json(silent=True) or {}
        P = data.get('matrix')
        pi0 = data.get('initial_state')
        steps = int(data.get('steps', 30))
        steps = max(1, min(steps, 200))
        state_names = data.get('state_names', None)

        if P is None or pi0 is None:
            return jsonify({'error': 'Missing required fields: matrix, initial_state'}), 400

        P_arr = np.array(P, dtype=float)
        n = P_arr.shape[0]

        # Validate
        if P_arr.shape[0] != P_arr.shape[1]:
            return jsonify({'error': 'Matrix must be square'}), 400
        if len(pi0) != n:
            return jsonify({'error': 'Initial state dimension mismatch'}), 400
        row_sums = P_arr.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=0.01):
            return jsonify({'error': f'Rows must sum to 1. Got: {row_sums.tolist()}'}), 400
        pi0_arr = np.array(pi0, dtype=float)
        if not np.isclose(pi0_arr.sum(), 1.0, atol=0.01):
            return jsonify({'error': f'Initial state must sum to 1. Got: {pi0_arr.sum():.3f}'}), 400

        history = simulate_markov(P, pi0, steps)
        steady = compute_steady_state(P)
        chart = generate_chart(history, steps, state_names)

        distributions = [h.tolist() for h in history]

        return jsonify({
            'chart': chart,
            'distributions': distributions,
            'steady_state': steady,
            'steps': steps,
            'n_states': n
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/tp2/exercise2', methods=['POST'])
def tp2_exercise2():
    try:
        data = request.get_json(silent=True) or {}
        lamb = float(data.get('lambda'))
        mu = float(data.get('mu'))
        servers = int(data.get('servers'))
        max_n = int(data.get('max_n', 12))
        max_n = max(0, min(max_n, 60))

        metrics = mm_s_metrics(lamb, mu, servers)
        if not metrics['stable']:
            return jsonify(metrics)

        probabilities = calculate_probabilities_mm_s(lamb, mu, servers, max_n)
        chart = generate_probability_bar_chart(probabilities)

        return jsonify({
            'metrics': metrics,
            'probabilities': probabilities,
            'chart': chart,
            'max_n': max_n
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/tp2/exercise1', methods=['POST'])
def tp2_exercise1():
    try:
        data = request.get_json(silent=True) or {}
        lamb = float(data.get('lambda'))
        mu = float(data.get('mu'))
        max_n = int(data.get('max_n', 12))
        max_n = max(0, min(max_n, 60))

        metrics = mm1_metrics(lamb, mu)
        if not metrics['stable']:
            return jsonify(metrics)

        probabilities = calculate_probabilities(lamb, mu, max_n)
        chart = generate_probability_bar_chart(probabilities)

        return jsonify({
            'metrics': metrics,
            'probabilities': probabilities,
            'chart': chart,
            'max_n': max_n
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/tp3/simulate', methods=['POST'])
def tp3_simulate():
    try:
        data = request.get_json(silent=True) or {}
        lamb = float(data.get('lambda'))
        mu = float(data.get('mu'))
        duration = float(data.get('duration'))

        N, Nq, T, Tq, Ts, SU = simulate_mm1(lamb, mu, duration, verbose=False)
        metrics = {
            'N': N,
            'Nq': Nq,
            'T': T,
            'Tq': Tq,
            'Ts': Ts,
            'SU': SU
        }
        chart = generate_tp3_metrics_chart(metrics)
        return jsonify({'metrics': metrics, 'chart': chart})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.errorhandler(404)
def not_found(_):
    return jsonify({'error': 'Route not found'}), 404

@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(_):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
