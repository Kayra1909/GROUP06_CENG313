# GROUP06_CENG313 — Pokémon TCG AI Battle Challenge (Playground)

**Group ID:** 06

| Member | Student ID |
|---|---|
| Enes İnançlı | 23118080051 |
| Berat Orkun Elibol | 23118080778 |
| Bengi Kayra Özdemir | 21118080409 |

**Kaggle competition:** <https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground>

**Current Kaggle score:** μ = **419.9** (13 Nov 2026, 22:29 TRT; started at 600.0, leaderboard #1 ≈ 1,589). Submission 1: heuristic baseline + sample deck, screenshot in `submissions/assignment1_score.png`. The rating keeps changing as ladder games are played.

## The problem in one paragraph

This is a *simulation* competition. We submit an agent that plays the Pokémon Trading Card Game in the official `cabt`
simulator; Kaggle makes it play best-of-three matches against other teams' agents and keeps a skill rating
N(μ, σ²) per submission (start μ₀ = 600, final ranking by a Bradley–Terry tournament). Offline we measure the win rate (with Wilson 95 % CIs)
against reference agents and a local Bradley–Terry rating.

## Repository layout

```
GROUP06_CENG313/
├── README.md
├── requirements.txt
├── .gitignore                      (excludes data/ and build/)
├── assignment1_eda_baseline.ipynb  EDA + baseline (this assignment)
├── assignment2_solution.ipynb      final solution (Assignment 2)
├── src/
│   ├── __init__.py
│   ├── data.py      card CSV loading & cleaning, engine tables, game simulation → tables
│   ├── models.py    agents (random / first / heuristic), evaluation, Bradley–Terry, win-probability model
│   └── utils.py     plotting, Wilson CI, Kaggle submission bundle
├── data/            competition data — NOT committed
└── submissions/     Kaggle score screenshots
```

## Getting the data

The competition data must not be committed (and card data must not be shared publicly — keep this repo private).

**Option A — Kaggle CLI** (needs `~/.kaggle/kaggle.json` and the competition rules accepted):

```bash
kaggle competitions download -c the-pokemon-company-ptcg-ai-battle-challenge-playground -p data
cd data && unzip the-pokemon-company-ptcg-ai-battle-challenge-playground.zip && cd ..
```

**Option B — website:** competition page → *Data* → *Download All*, then unzip the archive into `data/`.

After unzipping, `data/` contains `EN_Card_Data_R2_full.csv`, `sample_submission/.../deck.csv` and the simulator source.
The code finds the files recursively, so the nested folder names do not matter.

## Running the notebooks

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps kaggle-environments==1.33.3
jupyter notebook assignment1_eda_baseline.ipynb
```

`kaggle-environments` is installed with `--no-deps` on purpose: its full dependency list (JAX, OpenSpiel, transformers …)
is only needed for other games and does not install cleanly on Windows. The `cabt` simulator only needs `jsonschema` and
`requests`, which are in `requirements.txt`. The simulator ships compiled engines for Windows, macOS and Linux.

Run the notebook from the repository root (*Kernel → Restart & Run All*); it takes about 10 minutes because it plays
several thousand simulated games. Lower the `N_*_GAMES` constants in the first code cell for a quick run.

The notebook writes the Kaggle bundle to `build/submission.tar.gz` (git-ignored). Upload it on the competition page under
*Submit Agent*.

## Notes

* The simulator seeds itself from hardware entropy, so simulated numbers change slightly between runs; we always report
  confidence intervals.
* If GitHub does not render the notebook, open it on nbviewer: `https://nbviewer.org/github/ceng313-2026/GROUP06_CENG313/blob/main/assignment1_eda_baseline.ipynb` (works once the repository is visible to the viewer)
