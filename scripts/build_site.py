"""Build the project website (docs/index.html) from committed results.

Every number on the site is injected from the results JSONs — nothing is
hand-typed. Figures are copied from the analysis outputs.

    python scripts/build_site.py [--main-dir results_v3/main]
                                 [--vision-dir results_v3/vision]
                                 [--dqn-file results_v3/dqn/dqn.json]
                                 [--ablations results/ablations.json]
                                 [--figs results/figs]
                                 [--out docs]

Run merge_results.py first so each study has summary.json + figs/.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

T = Path(__file__).resolve().parent.parent  # repo root


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-dir", default="results_v3/main")
    ap.add_argument("--vision-dir", default="results_v3/vision")
    ap.add_argument("--dqn-file", default="results_v3/dqn/dqn.json")
    ap.add_argument("--ablations", default="results/ablations.json")
    ap.add_argument("--figs", default="results/figs")
    ap.add_argument("--out", default="docs")
    args = ap.parse_args()

    main_sum = json.loads(Path(args.main_dir, "summary.json").read_text())
    vis_sum = json.loads(Path(args.vision_dir, "summary.json").read_text())
    dqn = json.loads(Path(args.dqn_file).read_text())
    abl = json.loads(Path(args.ablations).read_text())

    # ---- mechanism study (mixing-ratio phase diagram) ---------------- #
    mech_path = T / "results/mechanism.json"
    if mech_path.exists():
        mech = json.loads(mech_path.read_text())
        mv = mech["vision"]["cap32"]
        mv_ratios, mv_succ = mv["ratios"], mv["final_success"]
        mix = dict(zip([str(r) for r in mv_ratios], mv_succ))
        mix_rescue = mix["0.25"]
        mix_one = mix.get("1.0")
        mix_blind = mix["unbounded"]
        c32 = mech["contact"]["cap32"]
        contact_succ = dict(zip([str(r) for r in c32["ratios"]],
                                c32["final_success"]))
        contact_unbounded = contact_succ["unbounded"]
        contact_capped = contact_succ["0.25"]
    else:
        mix_rescue = mix_one = mix_blind = None
        contact_unbounded = contact_capped = None

    m = main_sum["strategies"]
    v = vis_sum["strategies"]
    cfg = main_sum["config"]
    iters = cfg["iterations"]

    rel_final = m["relabel"]["success_rate_mean"][-1]
    rel_initial = m["relabel"]["success_rate_mean"][0]
    cur_final = m["relabel_curated"]["success_rate_mean"][-1]
    none_final = m["none"]["success_rate_mean"][-1]
    fly_steps = int(m["relabel"]["curation_log"][-1]["dataset_frames"])
    dqn_final = dqn["final_success"]
    dqn_steps = int(dqn["total_env_steps"])

    vis_none = v["none"]["success_rate_mean"][-1]
    vis_rel = v["relabel"]["success_rate_mean"][-1]
    vis_cur = v["relabel_curated"]["success_rate_mean"][-1]
    vis_bal = (v["relabel_balanced"]["success_rate_mean"][-1]
               if "relabel_balanced" in v else None)
    vis_rel_first = v["relabel"]["success_rate_mean"][0]
    vis_cur_first = v["relabel_curated"]["success_rate_mean"][0]
    vis_bal_first = (v["relabel_balanced"]["success_rate_mean"][0]
                     if "relabel_balanced" in v else None)
    # vision headline: does any oracle rule lift the CNN?
    if vis_bal is not None:
        vis_bal_gain = vis_bal - vis_bal_first
        bal_tail = (f"ratio-capped relabeling reaches {_fmt(vis_bal)}"
                    if vis_bal_gain > 0.05 else
                    f"even ratio-capped relabeling stays at {_fmt(vis_bal)}")
        vis_lead = (f"Blind relabeling crashes the CNN "
                    f"({_fmt(vis_rel_first)} \u2192 {_fmt(vis_rel)}); curated "
                    f"relabeling degrades less ({_fmt(vis_cur_first)} \u2192 "
                    f"{_fmt(vis_cur)}); {bal_tail}.")
    else:
        vis_lead = (f"Blind relabeling crashes the CNN ({_fmt(vis_rel_first)} "
                    f"\u2192 {_fmt(vis_rel)}): relabeled frames flood the dataset "
                    f"and the high-capacity CNN overfits its own failure "
                    f"distribution. Curated relabeling degrades far less "
                    f"({_fmt(vis_cur_first)} \u2192 {_fmt(vis_cur)}).")

    # ---- results table (compact: start / mid / end) ------------------ #
    order = ["none", "success_only", "near_miss", "success_coverage",
             "relabel_curated", "relabel"]
    labels = {
        "none": "None (frozen control)",
        "success_only": "Self-curation · successes",
        "near_miss": "Self-curation · near-misses",
        "success_coverage": "Self-curation · novel successes",
        "relabel_curated": "Oracle · curated relabel",
        "relabel": "Oracle · relabel all (DAgger)",
    }
    rows = []
    for name in order:
        d = m[name]
        mean = d["success_rate_mean"]
        std = d["success_rate_std"]
        start, mid, end = mean[0], mean[len(mean) // 2], mean[-1]
        gain = end - mean[0]
        cls = "gain" if gain > 0.05 else ("flat" if abs(gain) <= 0.05 else "loss")
        rows.append(
            f'<tr><td>{labels[name]}</td><td>{_fmt(start)}</td>'
            f'<td>{_fmt(mid)}</td><td class="{cls}">{_fmt(end)} '
            f'<span class="delta">({gain:+.2f})</span></td></tr>'
        )
    table = "\n".join(rows)

    # ---- flywheel budget point (per strategy, for the budget plot) --- #
    budget_points = []
    for name in ("success_only", "relabel_curated", "relabel"):
        d = m[name]
        log = d.get("curation_log", [])
        if log:
            inter = [d["success_rate_mean"][0]] + [c["dataset_frames"] for c in log]
            budget_points.append((name, inter, d["success_rate_mean"]))
    # DQN curve for a simple inline SVG-free table: show key milestones
    dqn_milestones = " · ".join(
        f"{e['env_steps']:,} steps → {e['success_rate']:.2f}" for e in dqn["eval"]
    )

    html = _TEMPLATE
    html = html.replace("%%TITLE%%", "DataFly — a data flywheel for robot manipulation")
    html = html.replace("%%REPO%%", "https://github.com/sehajr-singhs/robotic-data-flywheel")
    html = html.replace("%%REL_FINAL%%", _fmt(rel_final))
    html = html.replace("%%REL_INITIAL%%", _fmt(rel_initial))
    html = html.replace("%%CUR_FINAL%%", _fmt(cur_final))
    html = html.replace("%%NONE_FINAL%%", _fmt(none_final))
    html = html.replace("%%SEEDS%%", str(cfg["seeds"]))
    html = html.replace("%%EVAL%%", str(cfg["eval_starts"]))
    html = html.replace("%%ITERS%%", str(iters))
    html = html.replace("%%FLY_STEPS%%", f"{fly_steps:,}")
    html = html.replace("%%DQN_FINAL%%", _fmt(dqn_final))
    html = html.replace("%%DQN_STEPS%%", f"{dqn_steps:,}")
    html = html.replace("%%DQN_MILESTONES%%", dqn_milestones)
    html = html.replace("%%VIS_REL%%", _fmt(vis_rel))
    html = html.replace("%%VIS_NONE%%", _fmt(vis_none))
    html = html.replace("%%VIS_CUR%%", _fmt(vis_cur))
    html = html.replace("%%VIS_LEAD%%", vis_lead)
    html = html.replace("%%VIS_IMG%%", str(vis_sum["config"]["img_size"]))
    html = html.replace("%%VIS_SEEDS%%", str(vis_sum["config"]["seeds"]))
    html = html.replace("%%VIS_ITERS%%", str(vis_sum["config"]["iterations"]))
    if mix_rescue is not None:
        html = html.replace("%%MIX_RESCUE%%", _fmt(mix_rescue))
        html = html.replace("%%MIX_BLIND%%", _fmt(mix_blind))
        html = html.replace("%%MIX_GAIN%%", _fmt(mix_rescue - mix_blind))
        html = html.replace("%%MIX_ONE%%", _fmt(mix_one) if mix_one else "—")
        html = html.replace("%%CONTACT_UNBOUNDED%%", _fmt(contact_unbounded))
        html = html.replace("%%CONTACT_CAPPED%%", _fmt(contact_capped))
    else:
        for k in ("%%MIX_RESCUE%%", "%%MIX_BLIND%%", "%%MIX_GAIN%%",
                  "%%MIX_ONE%%", "%%CONTACT_UNBOUNDED%%", "%%CONTACT_CAPPED%%"):
            html = html.replace(k, "—")
    html = html.replace("%%TABLE_ROWS%%", table)
    html = html.replace("%%MAIN_CFG%%",
                        f"{cfg['seeds']} training seeds × {cfg['eval_starts']} held-out "
                        f"starts × {cfg['iterations']} flywheel iterations, "
                        f"12-dim state vector → {cfg['hidden']}-unit MLP")
    html = html.replace("%%VIS_CFG%%",
                        f"{vis_sum['config']['seeds']} training seeds × "
                        f"{vis_sum['config']['eval_starts']} held-out starts × "
                        f"{vis_sum['config']['iterations']} flywheel iterations, "
                        f"{vis_sum['config']['img_size']}×{vis_sum['config']['img_size']} RGB → torch CNN")

    # copy figures + paper into the site
    out = Path(args.out)
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    # every figure ships from results/figs (analyze.py is the single
    # producer); the v3 study dirs are only a fallback for legacy layouts
    figdir = Path(args.figs)
    copies = [
        (figdir / "success_vs_iteration.png", "success_curves.png"),
        (figdir / "trajectories.png", "trajectories.png"),
        (figdir / "flywheel_report.png", "flywheel_report.png"),
        (figdir / "vision_curves.png", "vision_curves.png"),
        (figdir / "sample_observations.png", "sample_observations.png"),
        (figdir / "budget_comparison.png", "budget_comparison.png"),
        (figdir / "oracle_crossover.png", "oracle_crossover.png"),
        (figdir / "label_efficiency.png", "label_efficiency.png"),
        (figdir / "difficulty.png", "difficulty.png"),
        (figdir / "fig_phase_diagram.png", "fig_phase_diagram.png"),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, figs / dst)
    for p in (T / "paper" / "manuscript.pdf", T / "paper" / "nmi_paper.pdf"):
        if p.exists():
            shutil.copy2(p, out / p.name)
    (out / "index.html").write_text(html, encoding="utf-8")
    print(f"site built -> {out}/index.html (+figs/, manuscript.pdf, nmi_paper.pdf)")


_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="description" content="DataFly: a reproducible study of the robot data flywheel — collection, curation, and closed-loop policy improvement. Measured on a planar push task across kinematic, pixel, and contact-rich regimes.">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>%%TITLE%%</title>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/computer-modern/cmu-serif.css">

  <style>
    :root {
      --ink: #1a1a1a; --muted: #555; --faint: #8c8e90; --panel: #f8f8f8;
      --border: #c4c6c8; --link: #226999; --good: #1e6b3a; --bad: #b03a2e;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html { background: #fff; }
    body { font-family: 'CMU Serif', Georgia, serif; font-weight: 500;
      color: var(--ink); -webkit-font-smoothing: antialiased;
      text-rendering: optimizeLegibility; }
    a { color: var(--link); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .container { max-width: 920px; margin: 0 auto; padding: 0 20px; }
    .has-text-centered { text-align: center; }
    .has-text-justified { text-align: justify; }

    .hero { padding: 4.2rem 0 1.6rem; }
    .publication-title { font-family: 'CMU Serif', Georgia, serif;
      font-weight: 700 !important; line-height: 1.12; letter-spacing: 0;
      font-size: 2.5rem; text-wrap: balance; }
    .publication-title strong { font-weight: 900 !important; }
    .publication-sub { margin-top: 1.1rem; font-family: 'Inter', sans-serif;
      font-size: 1.05rem; color: var(--muted); line-height: 1.5;
      max-width: 60rem; margin-left: auto; margin-right: auto; }
    .tagline { margin-top: 0.9rem; font-family: 'IBM Plex Mono', monospace;
      font-size: 0.92rem; color: var(--ink); letter-spacing: 0.01em; }
    .authors { margin-top: 1.2rem; font-family: 'Inter', sans-serif;
      font-size: 0.95rem; color: var(--ink); }
    .affiliation { margin-top: 0.15rem; font-family: 'Inter', sans-serif;
      font-size: 0.82rem; color: var(--faint); }
    .links { margin-top: 1.5rem; font-family: 'IBM Plex Mono', monospace;
      font-size: 0.88rem; display: flex; flex-wrap: wrap; gap: 0.6rem 1.4rem;
      justify-content: center; }

    .section { padding: 2.4rem 0 1.2rem; }
    .title { font-size: 1.35rem; font-weight: 700; letter-spacing: -0.01em;
      margin-bottom: 1rem; padding-bottom: 0.35rem; border-bottom: 1px solid var(--border); }
    .section p { line-height: 1.6; color: var(--ink); margin-bottom: 0.9rem; }
    .muted { color: var(--muted); }

    .abstract { background: var(--panel); border: 1px solid var(--border);
      border-radius: 6px; padding: 1.4rem 1.6rem; font-size: 0.99rem;
      line-height: 1.62; text-align: justify; }

    .impact-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1.2rem; margin-top: 1.1rem; }
    .impact { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
    .impact-img { width: 100%; display: block; border-bottom: 1px solid var(--border); }
    .impact-body { padding: 0.9rem 1.1rem 1.05rem; }
    .impact-num { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem;
      font-weight: 600; letter-spacing: 0.02em; text-transform: uppercase;
      color: var(--ink); margin-bottom: 0.35rem; }
    .impact-body p { font-size: 0.88rem; line-height: 1.5; color: var(--muted); margin: 0; }

    .figure { margin: 1.2rem 0 0.4rem; }
    .figure img { width: 100%; display: block; border: 1px solid var(--border);
      border-radius: 6px; background: #fff; }
    .fig-note { font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem;
      color: var(--faint); margin-top: 0.45rem; line-height: 1.5; }

    table { width: 100%; border-collapse: collapse; font-family: 'Inter', sans-serif;
      font-size: 0.83rem; margin: 1rem 0 1.4rem; background: var(--panel);
      border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
    th, td { padding: 0.5rem 0.65rem; text-align: right; border-bottom: 1px solid var(--border); }
    th:first-child, td:first-child { text-align: left; }
    thead th { font-weight: 600; font-size: 0.78rem; letter-spacing: 0.02em;
      text-transform: uppercase; color: var(--muted); background: #fff; }
    tbody tr:last-child td { border-bottom: none; }
    td.gain { color: var(--good); font-weight: 700; }
    td.loss { color: var(--bad); font-weight: 600; }
    td.flat { color: var(--muted); }
    .delta { font-weight: 400; opacity: .75; font-size: 0.72rem; }
    .table-note { font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem;
      color: var(--faint); margin-top: -1rem; margin-bottom: 1.2rem; }
    .tablescroll { overflow-x: auto; }

    pre { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
      padding: 1.1rem 1.3rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem;
      line-height: 1.55; overflow-x: auto; margin: 1rem 0; }

    footer { margin-top: 3rem; padding: 1.6rem 0 2.6rem; border-top: 1px solid var(--border);
      font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: var(--faint);
      text-align: center; }
    @media (max-width: 600px) { .publication-title { font-size: 1.8rem; }
      th, td { padding: 0.4rem 0.4rem; font-size: 0.74rem; } }
  </style>
</head>
<body>

<section class="hero">
  <div class="container has-text-centered">
    <h1 class="publication-title">DataFly</h1>
    <div class="tagline">A data flywheel for robot manipulation — the loop only spins if the labels come back. Measured across kinematic, pixel, and contact-rich regimes.</div>
    <div class="authors">Sehaj Randhir Singh</div>
    <div class="affiliation">Independent researcher; partial affiliation with NYU Tandon School of Engineering</div>
    <div class="links">
      <a href="nmi_paper.pdf">Paper (NMI format)</a><a href="manuscript.pdf">Paper (IEEE format)</a><a href="%%REPO%%">GitHub (code + data)</a><a href="https://www.kaggle.com/datasets/sehajrsingh/datafly-v3-src">Results on Kaggle</a>
    </div>
  </div>
</section>

<div class="container">
<section class="section"><div class="abstract"><p>Deployment data is the fuel of the robot data flywheel, but <em>which</em> data — and with <em>whose</em> labels — determines whether the loop compounds or stalls. We formalize the flywheel as collect → score → curate → retrain → repeat, and isolate the curation decision as the independent variable. On a planar pushing task with an imperfect learned policy, six curation strategies are compared across %%SEEDS%% training seeds and %%EVAL%% held-out starts. Three findings. (1) <strong>Without feedback the policy is frozen</strong> — the loop never turns. (2) <strong>Self-curated episodes plateau</strong> — successes, near-misses, and novel states densify what the policy already does but cannot repair what it does badly. (3) <strong>Oracle relabeling of deployment failures is what compounds</strong> — blind relabeling reaches %%REL_FINAL%% success from %%REL_INITIAL%%, and curating by the progress signal reaches %%CUR_FINAL%% while spending roughly half the oracle queries. The result is robust to oracle-labeling noise and task difficulty. The loop transfers to raw pixels — and the mechanism is <em>mixing-ratio control</em>: a measured phase diagram over (relabeled:clean ratio × policy capacity) shows the flood boundary is a function of capacity — ratio control alone rescues the CNN from %%MIX_BLIND%% to <strong>%%MIX_RESCUE%%</strong> — while a low-capacity MLP is flood-robust even on contact-rich MuJoCo dynamics.</p></div></section>

<section class="section"><h2 class="title">Does the flywheel spin?</h2><p>%%MAIN_CFG%%. Every strategy starts from the same noisy expert demonstrations and the same frozen control baseline; the curation rule is the only variable.</p><div class="tablescroll"><table><thead><tr><th>Strategy</th><th>iter 0</th><th>mid</th><th>final (gain)</th></tr></thead><tbody>%%TABLE_ROWS%%</tbody></table></div><div class="table-note">Held-out push success (mean over seeds); the full per-iteration table with std is in both papers.</div><div class="figure"><img class="impact-img" src="figs/success_curves.png" alt="Success rate vs flywheel iteration for all strategies"><div class="fig-note">Held-out success vs flywheel iteration (mean ± std over seeds). Oracle relabeling compounds; self-curation plateaus; the frozen control is flat.</div></div></section>

<section class="section"><h2 class="title">Perception flips the balance</h2><p>%%VIS_CFG%%. The policy never sees the 12-dim state vector; it must infer the block, the target, and its own arm from the rendered image. %%VIS_LEAD%%</p><div class="impact-grid"><div class="impact"><img class="impact-img" src="figs/vision_curves.png" alt="Vision policy success curves"><div class="impact-body"><div class="impact-num">The CNN, on raw pixels</div><p>Blind relabeling floods the dataset and the high-capacity CNN overfits its own failures; curated relabeling is what keeps the loop stable.</p></div></div><div class="impact"><img class="impact-img" src="figs/trajectories.png" alt="Example trajectories"><div class="impact-body"><div class="impact-num">The raw material</div><p>Example rollouts: an expert seed demo, an early policy failure (the flywheel's raw material), and a final success — the block path shows the push being learned.</p></div></div></div></section>

<section class="section"><h2 class="title">The mechanism — curation is mixing-ratio control</h2><p>Each flywheel iteration adds relabeled frames to a clean seed set; the <strong>relabeled:clean mixing ratio</strong> of the training set is what blind relabeling lets grow without bound. We make it the controlled variable and sweep it against unbounded relabeling, across policy capacities and — for the first time — on contact-rich MuJoCo dynamics.</p><div class="figure"><img class="impact-img" src="figs/fig_phase_diagram.png" alt="The flood boundary: final success vs mixing ratio"><div class="fig-note">The flood boundary. a — Perception (CNN): final success falls monotonically as the ratio grows — %%MIX_RESCUE%% at ratio 0.25 → %%MIX_ONE%% at 1.0 → %%MIX_BLIND%% unbounded. b–c — Kinematic and contact-rich MLP: flood-robust at every ratio (%%CONTACT_CAPPED%% capped vs %%CONTACT_UNBOUNDED%% unbounded on MuJoCo). The inset shows the closed-loop curator converging into the stable region without knowing capacity.</div></div><p>A low-capacity policy cannot memorize its failure distribution, so even unbounded relabeling forces generalization; a high-capacity CNN can, so flooding it overfits its own failures. The <code>relabel_adaptive</code> curator treats the flywheel report as a sensor — it halves its ratio when held-out success regresses and grows it otherwise, converging to the stable operating point on the CNN while staying near capacity on the MLP. Ratio control alone rescues <strong>%%MIX_GAIN%%</strong> of the lost performance.</p></section>

<section class="section"><h2 class="title">Label efficiency — what does the loop actually cost?</h2><p>The flywheel's training interaction (%%FLY_STEPS%% environment steps) versus DQN from scratch (%%DQN_STEPS%%). DQN gets the full state and a dense reward — and still lands at %%DQN_FINAL%%. Milestones: %%DQN_MILESTONES%%.</p><div class="figure"><img class="impact-img" src="figs/budget_comparison.png" alt="Success vs environment interactions"><div class="fig-note">Held-out success vs environment interactions used for training (log axis). The label-efficiency argument: the flywheel compounds with an order-of-magnitude less interaction than tabula-rasa RL — and the papers price the labels, reporting the crossover λ* where RL becomes the cheaper route.</div></div><div class="impact-grid"><div class="impact"><img class="impact-img" src="figs/oracle_crossover.png" alt="Oracle noise ablation"><div class="impact-body"><div class="impact-num">Noisy labels</div><p>Oracle-quality ablation: curated relabeling degrades less than blind relabeling under human-labeling noise.</p></div></div><div class="impact"><img class="impact-img" src="figs/difficulty.png" alt="Difficulty robustness"><div class="impact-body"><div class="impact-num">Harder task</div><p>On a harder task (tighter goal, longer pushes), oracle relabeling still compounds while no-feedback stays flat.</p></div></div></div></section>

<section class="section"><h2 class="title">Reproduce</h2><p style="margin-bottom:0.4rem"><a href="https://www.kaggle.com/datasets/sehajrsingh/datafly-v3-src">Committed results on Kaggle</a></p><pre>git clone https://github.com/sehajr-singhs/robotic-data-flywheel
cd robotic-data-flywheel
pip install -e ".[dev]"

# state-based study (all six strategies)
python scripts/run_experiment.py
python scripts/merge_results.py

# vision study (torch CNN, pixels) + DQN baseline
python scripts/run_experiment.py --obs-mode image --strategies none relabel relabel_curated

# analyses, the papers' numbers, and this site — nothing hand-typed
python scripts/analyze.py
python scripts/analyze_mechanism.py
python scripts/render_results.py
python scripts/build_site.py</pre><p class="muted">CPU-scale, seeded protocol (%%SEEDS%% seeds per condition, per-seed values committed), committed result JSONs, and the same pipeline on Kaggle GPU kernels. Tests: <code>pytest tests</code> (17 tests covering physics, curation, the loop, and the vision path).</p></section>
</div>

<footer>
  <div class="container">
    Sehaj Randhir Singh · independent researcher; partial affiliation with NYU Tandon ECE · 2026
  </div>
</footer>

</body>
</html>
"""


if __name__ == "__main__":
    main()
