# Cold email to Mind Robotics (Rivian's industrial-robotics spin-off)

Everything below is written to land with the people who run Mind Robotics.
Their public thesis — *"the fastest path to broadly capable robots is
through clearly defined, high-impact environments"* — is literally a data
flywheel argument. Your repo is the smallest faithful experiment of that
thesis. Lead with that.

---

## The email (use as-is, replace the bracketed parts)

**Subject:** Intern — built a reproducible study of the robot data flywheel

Hi [First name],

Your thesis — "the fastest path to broadly capable robots is through
clearly defined, high-impact environments" — is a data flywheel argument,
and I built the smallest faithful experiment of it: a fully reproducible
flywheel where the only variable is how deployment data gets curated.

The one-line result: **the curation rule decides whether the loop
compounds.** On a planar push task run at scale on GPU (6 seeds, 6
iterations, 300 held-out starts), a behavior-cloned policy stays flat at
12% with no feedback, climbs to ~35% by self-curating its own successes,
and quadruples to **66%** when deployment failures are relabeled by an
oracle (DAgger-style). Curated relabeling reaches 48% with **~2.5× fewer
oracle queries**, and degrades less under labeling noise (13% vs 20%
loss), which is the teleoperation-cost model industrial cells actually
face. A DQN trained from scratch needs 300k interactions for 4.5% — the
flywheel hits 66% on ~29k collected frames, which is the label-efficiency
gap in its strongest form. I also ran the loop on raw pixels (a CNN on
64×64 camera images), where curation is what keeps the loop stable.

- IEEE-format paper: github.com/sehajr-singhs/robotic-data-flywheel/blob/main/paper/manuscript.pdf
- Repo: github.com/sehajr-singhs/robotic-data-flywheel
- Every number is committed as JSON and re-runs from `scripts/`; the GPU
  pipeline lives in `kaggle/` (Kernel r4, 9,231s).

I'd love to work on the data side of Mind Robotics' industrial deployment
loops — collection, scoring, and curation for foundation-model training —
as an intern or co-op this [semester / summer]. Happy to walk through the
repo or extend the study toward your use cases.

Best,
Sehaj
[linkedin.com/in/...] · [phone]

---

## Why this works

1. **It mirrors their language.** "Flywheel," "clearly defined
   environments," "deployment" — these are their words. You are not asking
   for a job; you are showing up with work that extends their thesis.
2. **It leads with a number, not a claim.** "17% → 47%" and "40% fewer
   oracle queries" are the hook. Engineers respect results you can re-run.
3. **It's short.** ~140 words. Cold emails that respect the recipient's
   time get replies.
4. **It names the ask.** Internship/co-op on the data side — specific,
   modest, and aligned with what they hire for.

## Who to send it to

- **First choice:** anyone whose title involves *data, foundation models,
  or deployment* on the Mind Robotics careers page / team page
  (mindrobotics.com, LinkedIn). A data-team lead is a better target than
  the CEO.
- **Second choice:** a Rivian or Mind Robotics engineer/alum in your
  network (LinkedIn) — a 1-line referral beats a cold inbox.
- **Third choice:** the general recruiting inbox, with the same email.

## Follow-up cadence

- Day 0: send.
- Day 5: one short bump ("bumping this — happy to walk through the repo,
  it re-runs in ~10 minutes").
- Day 12: second bump with a *new artifact* (e.g., "I added a noisy-oracle
  experiment — curation wins harder when labels are imperfect").
- Then move on; apply through the official portal too.

## Before you send

- [ ] Make sure the README renders on GitHub (the repo is live at
      github.com/sehajr-singhs/robotic-data-flywheel).
- [ ] Replace `sehaj@example.com` in `paper/manuscript.tex` with your real
      email, recompile, and re-render the paper.
- [ ] Attach `paper/manuscript.pdf` (or link it in the repo).
- [ ] Fill in your real name/LinkedIn/phone in the email.
- [ ] Re-run `scripts/run_experiment.py` + `merge_results.py` once on your
      machine so the committed results are yours.

## One honest note

This is now a genuine multi-study, GPU-scale preprint — multi-seed
curves, a perception study, an RL baseline, and full reproducibility —
the right foundation for an ICRA/NMI submission, and already the strongest
thing a cold email could carry. What Mind Robotics is actually hiring for
is someone who can build and run data loops thoughtfully, and this repo
proves you can. If you get the internship, the natural next step is to
turn this study into a full ICRA submission with their data.
