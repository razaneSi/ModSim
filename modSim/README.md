# MarkovSim

Markov Chain Simulation Dashboard — Flask + NumPy + Matplotlib

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000

## Structure

```
markovsim/
├── app.py                  # Flask backend
├── templates/
│   └── index.html          # Frontend (HTML/CSS/JS)
├── static/
│   └── background.jpg      # Background image
└── requirements.txt
```

## Features
- Dynamic N×N transition matrix editor (2–6 states)
- Real-time probability evolution charts (Matplotlib)
- Automatic steady-state computation (NumPy lstsq)
- Built-in exercise presets (TP1 exercises)
- Step-by-step distribution viewer
