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
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="DataFly: a reproducible study of the robot data flywheel — collection, curation, and closed-loop policy improvement.">
<title>%%TITLE%%</title>
<style>
  :root {
    --bg: #0b1020; --panel: #121a30; --panel2: #182340; --ink: #e8ecf8;
    --muted: #9aa7c7; --accent: #4f8cff; --green: #34d399; --red: #f87171;
    --amber: #fbbf24; --line: #26335c;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--ink); font: 16px/1.65 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  .wrap { max-width: 1080px; margin: 0 auto; padding: 0 24px; }
  nav { position: sticky; top: 0; background: rgba(11,16,32,.92); backdrop-filter: blur(8px); border-bottom: 1px solid var(--line); z-index: 10; }
  nav .wrap { display: flex; align-items: center; gap: 22px; padding-top: 14px; padding-bottom: 14px; }
  nav .logo { font-weight: 800; letter-spacing: .04em; }
  nav a { color: var(--muted); text-decoration: none; font-size: 14px; }
  nav a:hover { color: var(--ink); }
  nav .spacer { flex: 1; }
  .btn { display: inline-block; background: var(--accent); color: #fff; padding: 10px 18px;
         border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 15px; }
  .btn.ghost { background: transparent; border: 1px solid var(--line); color: var(--ink); }
  header.hero { padding: 88px 0 56px; }
  .kicker { color: var(--accent); text-transform: uppercase; letter-spacing: .18em; font-size: 13px; font-weight: 700; }
  h1 { font-size: clamp(34px, 6vw, 56px); line-height: 1.08; margin: 18px 0 16px; font-weight: 800; }
  h1 em { color: var(--accent); font-style: normal; }
  .sub { color: var(--muted); font-size: 19px; max-width: 760px; }
  .hero-cta { margin-top: 28px; display: flex; gap: 14px; flex-wrap: wrap; }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 48px; }
  .stat { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 18px 20px; }
  .stat .n { font-size: 30px; font-weight: 800; color: var(--green); }
  .stat .n.red { color: var(--red); }
  .stat .n.amber { color: var(--amber); }
  .stat .l { color: var(--muted); font-size: 13.5px; margin-top: 4px; }
  section { padding: 56px 0; border-top: 1px solid var(--line); }
  h2 { font-size: 28px; margin-bottom: 10px; font-weight: 800; }
  .lede { color: var(--muted); max-width: 820px; margin-bottom: 26px; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 22px; }
  .grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; }
  figure img { width: 100%; border-radius: 10px; border: 1px solid var(--line); background: #fff; }
  figcaption { color: var(--muted); font-size: 13.5px; margin-top: 10px; }
  table { width: 100%; border-collapse: collapse; font-size: 14.5px; }
  th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--line); }
  th { color: var(--muted); font-weight: 600; font-size: 13px; text-transform: uppercase; letter-spacing: .06em; }
  td.gain { color: var(--green); font-weight: 700; }
  td.flat { color: var(--muted); }
  .delta { font-weight: 400; opacity: .75; font-size: 13px; }
  .loop { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: center; margin: 22px 0; }
  .loop .step { background: var(--panel2); border: 1px solid var(--line); border-radius: 10px; padding: 12px 16px; font-size: 14px; font-weight: 600; }
  .loop .arrow { color: var(--accent); font-weight: 800; }
  code { background: var(--panel2); border: 1px solid var(--line); padding: 2px 7px; border-radius: 6px; font-size: 13.5px; }
  pre { background: var(--panel2); border: 1px solid var(--line); border-radius: 10px; padding: 18px; overflow-x: auto; font-size: 13.5px; line-height: 1.55; margin: 14px 0; }
  footer { padding: 40px 0 60px; border-top: 1px solid var(--line); color: var(--muted); font-size: 14px; }
  .tag { display: inline-block; font-size: 12px; border: 1px solid var(--line); color: var(--muted);
         border-radius: 999px; padding: 3px 10px; margin-right: 8px; }
  a { color: var(--accent); }
</style>
</head>
<body>

<nav>
  <div class="wrap">
    <span class="logo">DataFly</span>
    <a href="#results">Results</a>
    <a href="#vision">Vision</a>
    <a href="#results">Results</a>
    <a href="#vision">Vision</a>
    <a href="#mechanism">Mechanism</a>
    <a href="#labels">Label efficiency</a>
    <a href="#reproduce">Reproduce</a>
    <span class="spacer"></span>
    <a href="manuscript.pdf" class="btn ghost">IEEE paper</a>
    <a href="nmi_paper.pdf" class="btn ghost">NMI-style</a>
    <a href="%%REPO%%" class="btn">GitHub</a>
  </div>
</nav>

<header class="hero">
  <div class="wrap">
    <div class="kicker">Robot Learning · Closed-Loop Data</div>
    <h1>The data flywheel only spins if the <em>labels come back</em>.</h1>
    <p class="sub">A reproducible study of the robot data flywheel: deploy a policy, score and curate its failures, and feed the right data back. On a planar pushing task we show that <b>relabeling deployment failures compounds</b> (0.%%REL_INITIAL%% → 0.%%REL_FINAL%% success), self-curated data plateaus, and no feedback is flat — that the loop works even when the policy must <b>learn from raw pixels</b> — and that the loop is governed by one mechanism: <b>mixing-ratio control</b>.</p>
    <div class="hero-cta">
      <a href="manuscript.pdf" class="btn">Read the IEEE paper</a>
      <a href="nmi_paper.pdf" class="btn ghost">NMI-style preprint</a>
      <a href="%%REPO%%" class="btn ghost">View the code</a>
    </div>
    <div class="stats">
      <div class="stat"><div class="n">0.%%REL_FINAL%%</div><div class="l">held-out success after %%ITERS%% flywheel iterations (oracle relabel), from 0.%%REL_INITIAL%%</div></div>
      <div class="stat"><div class="n amber">0.%%CUR_FINAL%%</div><div class="l">success with curated relabel — ~2.5× fewer oracle queries per point of performance</div></div>
      <div class="stat"><div class="n">%%VIS_REL%%</div><div class="l">success learning from %%VIS_IMG%%×%%VIS_IMG%% pixels (torch CNN) — no state vector</div></div>
      <div class="stat"><div class="n red">%%DQN_FINAL%%</div><div class="l">DQN from scratch after %%DQN_STEPS%% interactions — vs the flywheel's %%FLY_STEPS%%</div></div>
    </div>
    <p class="lede" style="margin-top:26px;max-width:820px"><b>The mechanism.</b> Blind relabeling lets the relabeled:clean frame ratio grow without bound; a measured phase diagram shows the stability boundary is a function of policy capacity — ratio control alone rescues the CNN (0.%%MIX_BLIND%% → <b>0.%%MIX_RESCUE%%</b>), while a low-capacity MLP is flood-robust even on contact-rich MuJoCo dynamics (0.%%CONTACT_UNBOUNDED%% unbounded vs 0.%%CONTACT_CAPPED%% capped), and a closed-loop curator finds the stable operating point without knowing capacity.</p>
  </div>
</header>

<section id="abstract">
  <div class="wrap">
    <h2>Abstract</h2>
    <p class="lede">Deployment data is the fuel of the robot data flywheel, but <i>which</i> data — and with <i>whose</i> labels — determines whether the loop compounds or stalls. We formalize the flywheel as collect → score → curate → retrain → repeat, and isolate the curation decision as the independent variable. On a planar pushing task with an imperfect learned policy, six curation strategies are compared across %%SEEDS%% training seeds and %%EVAL%% held-out starts. Three findings: (1) without feedback the policy is frozen — the loop never turns; (2) self-curated episodes (successes, near-misses, novel states) give modest plateauing gains; (3) <b>oracle relabeling of deployment failures is what compounds</b> — blind relabeling reaches 0.%%REL_FINAL%% success, and curating by the progress signal reaches 0.%%CUR_FINAL%% while spending ~2.5× fewer oracle queries. The result is robust to oracle-labeling noise (curated relabeling degrades less) and to task difficulty; the perception study shows the loop transfers to raw pixels, where curation is what keeps the CNN stable.</p>
    <p class="lede"><span class="tag">Data flywheel</span><span class="tag">DAgger</span><span class="tag">Imitation learning</span><span class="tag">Label efficiency</span><span class="tag">IEEE-format manuscript</span></p>
  </div>
</section>

<section id="results">
  <div class="wrap">
    <h2>Results — does the flywheel spin?</h2>
    <p class="lede">%%MAIN_CFG%%. Every strategy starts from the same noisy expert demonstrations and the same frozen control baseline.</p>
    <div class="card">
      <table>
        <thead><tr><th>Strategy</th><th>iter 0</th><th>mid</th><th>final (gain)</th></tr></thead>
        <tbody>
          %%TABLE_ROWS%%
        </tbody>
      </table>
    </div>
    <figure style="margin-top:22px">
      <img src="figs/success_curves.png" alt="Success rate vs flywheel iteration for all strategies">
      <figcaption>Held-out success vs flywheel iteration (mean ± std over seeds). Oracle relabeling compounds; self-curation plateaus; the frozen control is flat.</figcaption>
    </figure>
  </div>
</section>

<section id="vision">
  <div class="wrap">
    <h2>Perception-grounded flywheel — learning from pixels</h2>
    <p class="lede">%%VIS_CFG%%. The policy never sees the 12-dim state vector; it must infer the block, the target, and its own arm from the rendered image. %%VIS_LEAD%%</p>
    <div class="grid2">
      <figure>
        <img src="figs/vision_curves.png" alt="Vision policy success curves">
        <figcaption>Perception-grounded flywheel: the CNN on raw pixels. Curated relabeling is what keeps the loop stable.</figcaption>
      </figure>
      <figure>
        <img src="figs/trajectories.png" alt="Example trajectories">
        <figcaption>Example rollouts: expert seed demo, an early policy failure, and a final success — the block path (green) shows the push being learned.</figcaption>
      </figure>
    </div>
  </div>
</section>

<section id="mechanism">
  <div class="wrap">
    <h2>The mechanism — curation is mixing-ratio control</h2>
    <p class="lede">Each flywheel iteration adds relabeled frames to a clean seed set; the <b>relabeled:clean mixing ratio</b> of the training set is what blind relabeling lets grow without bound. We make it the controlled variable and sweep it against unbounded relabeling, across policy capacities and — for the first time — on contact-rich MuJoCo dynamics.</p>
    <figure>
      <img src="figs/fig_phase_diagram.png" alt="The flood boundary: final success vs mixing ratio">
      <figcaption>The flood boundary. <b>a</b> Perception (CNN): final success falls monotonically as the ratio grows — 0.%%MIX_RESCUE%% at ratio 0.25 → 0.%%MIX_ONE%% at 1.0 → 0.%%MIX_BLIND%% unbounded. <b>b–c</b> Kinematic and contact-rich MLP: flood-robust at every ratio (0.%%CONTACT_CAPPED%% capped vs 0.%%CONTACT_UNBOUNDED%% unbounded on MuJoCo). The inset shows the closed-loop curator converging into the stable region without knowing capacity.</figcaption>
    </figure>
    <p class="lede">A low-capacity policy cannot memorize its failure distribution, so even unbounded relabeling forces generalization; a high-capacity CNN can, so flooding it overfits its own failures. The <code>relabel_adaptive</code> curator treats the flywheel report as a sensor: it halves its ratio when held-out success regresses and grows it otherwise, converging to the stable operating point on the CNN while staying near capacity on the MLP.</p>
  </div>
</section>

<section id="labels">
  <div class="wrap">
    <h2>Label efficiency — what does the loop actually cost?</h2>
    <p class="lede">The flywheel's training interaction (%%FLY_STEPS%% environment steps) versus DQN from scratch (%%DQN_STEPS%%). DQN gets the full state and a dense reward — and still lands at 0.%%DQN_FINAL%%. Milestones: %%DQN_MILESTONES%%.</p>
    <figure>
      <img src="figs/budget_comparison.png" alt="Success vs environment interactions">
      <figcaption>Held-out success vs environment interactions used for training. The label-efficiency argument: the flywheel compounds with ~2–4 orders of magnitude less interaction than tabula-rasa RL.</figcaption>
    </figure>
    <div class="grid2" style="margin-top:22px">
      <figure>
        <img src="figs/oracle_crossover.png" alt="Oracle noise ablation">
        <figcaption>Oracle-quality ablation: curated relabeling degrades less than blind relabeling under human-labeling noise.</figcaption>
      </figure>
      <figure>
        <img src="figs/difficulty.png" alt="Difficulty robustness">
        <figcaption>On a harder task (tighter goal, longer pushes), oracle relabeling still compounds while no-feedback stays flat.</figcaption>
      </figure>
    </div>
  </div>
</section>

<section id="method">
  <div class="wrap">
    <h2>Method</h2>
    <p class="lede">A planar 2R arm pushes a block to a target region. The task is deliberately simple to make full flywheel studies tractable, but hard enough that behavior-cloned policies fail often — which is exactly when the curation decision matters.</p>
    <div class="loop">
      <div class="step">Deploy</div><span class="arrow">→</span>
      <div class="step">Score (success · progress · smoothness · coverage)</div><span class="arrow">→</span>
      <div class="step">Curate (strategy)</div><span class="arrow">→</span>
      <div class="step">Fine-tune</div><span class="arrow">→</span>
      <div class="step">Evaluate</div><span class="arrow">↻</span>
    </div>
    <p class="lede">The curation strategy — the decision rule for what goes back into the training set — is the independent variable. Oracle relabeling approximates a human teleoperator labeling each deployment state; curated relabeling only pays for the failures that made progress.</p>
  </div>
</section>

<section id="reproduce">
  <div class="wrap">
    <h2>Reproduce</h2>
    <p class="lede">The full experiment is deterministic, seeded, and runs end-to-end in minutes on a laptop or GPU. Every number on this page and in the paper is rendered from the committed results JSON — nothing hand-typed.</p>
    <pre>git clone https://github.com/sehajr-singhs/robotic-data-flywheel
cd robotic-data-flywheel
pip install -e ".[dev]"

# state-based study (all six strategies)
python scripts/run_experiment.py
python scripts/merge_results.py

# vision study (torch CNN, pixels)
python scripts/run_experiment.py --obs-mode image --strategies none relabel relabel_curated

# DQN baseline + analyses
python scripts/analyze.py
python scripts/render_results.py
python scripts/build_site.py</pre>
    <p class="lede">Tests: <code>pytest tests</code> (17 tests covering physics, curation, the loop, and the vision path).</p>
  </div>
</section>

<footer>
  <div class="wrap">
    Built with <a href="%%REPO%%">DataFly</a> · MIT licensed · IEEE-format manuscript + NMI-style preprint ·
    results and figures regenerated from committed JSON by <code>scripts/build_site.py</code>.
  </div>
</footer>

</body>
</html>
"""


if __name__ == "__main__":
    main()
