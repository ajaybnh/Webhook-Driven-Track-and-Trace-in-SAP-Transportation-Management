"""Discrete-event simulation of webhook-driven track and trace ingestion.

Experiments
  E1  ordering strategies (arrival order, hold window, post-immediately + monotonic)
  E2  business timestamp source (sender event time vs receipt time)
  E3  duplicate handling (none, non-atomic check, atomic check) and fan-out keys
  E4  matching under equipment reuse and early events

All times are in minutes. Each replication simulates N freight orders (FOs),
each with four expected events: departure from origin (1), arrival at
destination (2), unload end (3), proof of delivery (4).
"""
import json
import sys
import numpy as np

N_FO = 10_000
REPS = 20
SEED0 = 20261021

P = dict(
    lat_median_min=20 / 60,      # normal delivery latency median 20 s
    lat_sigma=1.0,               # lognormal sigma
    p_hold=0.10,                 # share of events held in a coverage gap
    hold_lo=5, hold_hi=90,       # gap hold duration, minutes
    p_dup=0.03,                  # share of messages the sender delivers twice
    p_fast_retry=0.40,           # share of duplicates that are near-simultaneous
    fast_hi=3 / 60,              # near-simultaneous retry gap upper bound (3 s)
    slow_lo=1, slow_hi=360,      # later retry gap, minutes
    proc=2 / 60,                 # receiver check-to-commit time (2 s)
    plan_sd=25,                  # actual minus planned, sd minutes
    late_tol=30,                 # late alert tolerance, minutes
    horizon=240,                 # observation after final event, minutes
)

RANKS = 4


def gen_orders(rng, p, n=N_FO):
    start = rng.uniform(0, 30 * 1440, n)
    g1 = rng.uniform(120, 480, n)   # travel
    g2 = rng.uniform(30, 180, n)    # dwell to unload end
    g3 = rng.uniform(5, 30, n)      # unload end to POD
    plan = np.stack([start, start + g1, start + g1 + g2, start + g1 + g2 + g3], 1)
    noise = rng.normal(0, p["plan_sd"], (n, 4))
    actual = plan + noise
    actual = np.maximum.accumulate(actual, axis=1)  # keep real-world order
    # strictly increasing
    actual[:, 1:] = np.maximum(actual[:, 1:], actual[:, :-1] + 0.5)
    return plan, actual


def gen_receipts(rng, p, actual):
    n = actual.shape[0]
    mu = np.log(p["lat_median_min"])
    lat = rng.lognormal(mu, p["lat_sigma"], (n, 4))
    hold = rng.random((n, 4)) < p["p_hold"]
    lat = lat + hold * rng.uniform(p["hold_lo"], p["hold_hi"], (n, 4))
    return actual + lat


# ---------------------------------------------------------------- E1 ordering
def display_timeline(strategy, rec, act, W=None):
    """Return list of (time, displayed_rank) changes for one FO."""
    order = np.argsort(rec, kind="stable")
    tl = []
    if strategy == "arrival":
        for k in order:
            tl.append((rec[k], k + 1))
    elif strategy == "monotonic":
        cur = 0
        for k in order:
            if k + 1 > cur:
                cur = k + 1
            tl.append((rec[k], cur))
    elif strategy == "hold":
        # reorder buffer: an event is due at receipt + W. When the first
        # event becomes due, every buffered event is flushed in event-time
        # order. Events received after a flush cannot be reordered with it.
        pending = sorted(range(4), key=lambda k: rec[k])
        while pending:
            due = rec[pending[0]] + W
            batch = [k for k in pending if rec[k] <= due]
            pending = [k for k in pending if rec[k] > due]
            for k in sorted(batch, key=lambda k: act[k]):
                tl.append((due, k + 1))
    return tl


def mismatch_and_metrics(tl, act):
    """Time-weighted share where display != true status, regressions,
    and per-milestone visibility latency."""
    t0, t1 = act[0], act[3] + P["horizon"]
    # build display step function
    tl = sorted(tl, key=lambda x: x[0])  # stable for equal times
    disp_pts = [(t0, 0)]
    for t, r in tl:
        disp_pts.append((t, r))
    regress = 0
    last = 0
    for _, r in tl:
        if r < last:
            regress += 1
        last = r
    # breakpoints
    bps = sorted(set([t0, t1] + [t for t, _ in tl if t0 < t < t1] + [a for a in act if t0 < a < t1]))

    def disp_at(t):
        r = 0
        for tt, rr in tl:
            if tt <= t:
                r = rr
            else:
                break
        return r

    def true_at(t):
        return int(np.sum(act <= t))

    mis = 0.0
    for a, b in zip(bps[:-1], bps[1:]):
        m = (a + b) / 2
        if disp_at(m) != true_at(m):
            mis += b - a
    share = mis / (t1 - t0)
    # visibility latency per milestone k: first time display >= k
    vis = []
    for k in range(1, 5):
        first = next((t for t, r in tl if r >= k), np.nan)
        vis.append(first - act[k - 1])
    return share, regress, vis


def run_e1(p, reps=REPS, n=N_FO):
    strategies = [("arrival", None), ("hold", 10), ("hold", 30), ("monotonic", None)]
    out = {f"{s}{'' if w is None else '_' + str(w)}": dict(share=[], regress=[], vis_med=[], vis_p95=[], inv=[])
           for s, w in strategies}
    for r in range(reps):
        rng = np.random.default_rng(SEED0 + r)
        plan, act = gen_orders(rng, p, n)
        rec = gen_receipts(rng, p, act)
        inv = float(np.mean(np.any(np.diff(rec, axis=1) < 0, axis=1)))
        for s, w in strategies:
            key = f"{s}{'' if w is None else '_' + str(w)}"
            shares, regs, viss = [], 0, []
            for i in range(n):
                tl = display_timeline(s, rec[i], act[i], w)
                sh, rg, vis = mismatch_and_metrics(tl, act[i])
                shares.append(sh)
                regs += rg
                viss.extend(vis)
            viss = np.array(viss)
            out[key]["share"].append(float(np.mean(shares)) * 100)
            out[key]["regress"].append(regs / n * 1000)
            out[key]["vis_med"].append(float(np.nanmedian(viss)))
            out[key]["vis_p95"].append(float(np.nanpercentile(viss, 95)))
            out[key]["inv"].append(inv * 100)
    return out


# ---------------------------------------------------------- E2 timestamp source
def run_e2(p, reps=REPS, n=N_FO):
    res = dict(event=dict(err_med=[], false_alerts=[], true_alerts=[]),
               receipt=dict(err_med=[], err_p95=[], false_alerts=[], true_alerts=[]))
    for r in range(reps):
        rng = np.random.default_rng(SEED0 + 1000 + r)
        plan, act = gen_orders(rng, p, n)
        rec = gen_receipts(rng, p, act)
        tol = p["late_tol"]
        truly_late = (act - plan) > tol
        # event time stamping: recorded == actual
        res["event"]["err_med"].append(0.0)
        res["event"]["false_alerts"].append(0.0)
        res["event"]["true_alerts"].append(float(truly_late.sum()) / n * 1000)
        err = rec - act
        alert = (rec - plan) > tol
        res["receipt"]["err_med"].append(float(np.median(err)))
        res["receipt"]["err_p95"].append(float(np.percentile(err, 95)))
        res["receipt"]["false_alerts"].append(float((alert & ~truly_late).sum()) / n * 1000)
        res["receipt"]["true_alerts"].append(float((alert & truly_late).sum()) / n * 1000)
    return res


# ------------------------------------------------------ E3 duplicates, fan-out
def run_e3(p, reps=REPS, n_events=40_000):
    res = dict(none=[], nonatomic=[], atomic=[])
    fan = dict(msg_after_success=dict(dup=[], lost=[]),
               msg_on_receipt=dict(dup=[], lost=[]),
               per_target=dict(dup=[], lost=[]))
    for r in range(reps):
        rng = np.random.default_rng(SEED0 + 2000 + r)
        dup = rng.random(n_events) < p["p_dup"]
        fast = rng.random(n_events) < p["p_fast_retry"]
        gap = np.where(fast, rng.uniform(0, p["fast_hi"], n_events),
                       rng.uniform(p["slow_lo"], p["slow_hi"], n_events))
        res["none"].append(dup.sum() / n_events * 1000)
        res["nonatomic"].append((dup & (gap < p["proc"])).sum() / n_events * 1000)
        res["atomic"].append(0.0)
        # fan-out: each trailer event targets k FOs; each target update fails
        # with probability pf on an attempt; sender/gateway retries the whole
        # message up to 3 times.
        k = rng.choice([1, 2, 3, 4], n_events, p=[0.4, 0.3, 0.2, 0.1])
        pf = 0.01
        d_a = l_a = d_b = l_b = d_c = l_c = 0
        targets_total = int(k.sum())
        for kk in k:
            # attempt 1
            ok1 = rng.random(kk) >= pf
            if ok1.all():
                continue
            # A: key written only after full success -> retry reposts the
            #    targets that already succeeded
            done = ok1.copy()
            for _ in range(3):
                okr = rng.random(kk) >= pf
                d_a += int((done & okr).sum())  # reposted to already-updated
                done = done | okr
                if okr.all():
                    break
            l_a += int((~done).sum())
            # B: key written on receipt -> retry is discarded as duplicate,
            #    failed targets are never updated
            l_b += int((~ok1).sum())
            # C: key per (message, target) -> retry touches only failed ones
            done = ok1.copy()
            for _ in range(3):
                okr = rng.random(kk) >= pf
                done = done | okr
                if done.all():
                    break
            l_c += int((~done).sum())
        fan["msg_after_success"]["dup"].append(d_a / targets_total * 1000)
        fan["msg_after_success"]["lost"].append(l_a / targets_total * 1000)
        fan["msg_on_receipt"]["dup"].append(0.0)
        fan["msg_on_receipt"]["lost"].append(l_b / targets_total * 1000)
        fan["per_target"]["dup"].append(0.0)
        fan["per_target"]["lost"].append(l_c / targets_total * 1000)
    return res, fan


# ------------------------------------------------------------ E4 matching
def run_e4(p, reps=REPS, n=N_FO, n_trailers=400, p_open=0.10, close_lo=1440, close_hi=14 * 1440, p_early=0.02,
           win_before=360, win_after=720, retry_hold=120):
    names = ["equipment_only", "equipment_window", "equipment_window_stage"]
    res = {k: dict(wrong=[], first_pass=[]) for k in names}
    early = dict(discard_lost=[], hold_lost=[])
    for r in range(reps):
        rng = np.random.default_rng(SEED0 + 3000 + r)
        plan, act = gen_orders(rng, p, n)
        order = np.argsort(plan[:, 0])
        plan, act = plan[order], act[order]
        trailer = np.zeros(n, int)
        free_at = np.zeros(n_trailers)
        for i in range(n):
            # pick the trailer that has been free longest
            j = int(np.argmin(free_at))
            trailer[i] = j
            free_at[j] = max(free_at[j], act[i, 3] + rng.uniform(60, 1440))
        # stale: FO whose POD was never reported, so it stays open
        stale = rng.random(n) < p_open
        # housekeeping closes a stale FO 1 to 14 days after its planned end
        open_until = plan[:, 3] + rng.uniform(close_lo, close_hi, n)
        last_on_trailer = {}
        prev = np.full(n, -1)
        for i in range(n):
            j = trailer[i]
            if j in last_on_trailer:
                prev[i] = last_on_trailer[j]
            last_on_trailer[j] = i
        wrong = {k: 0 for k in names}
        events = 0
        for i in range(n):
            # open earlier FOs on the same trailer: walk back the chain
            cands = []
            q = prev[i]
            depth = 0
            while q >= 0 and depth < 50:
                if stale[q]:
                    cands.append(q)
                q = prev[q]
                depth += 1
            for k in range(4):
                if stale[i] and k == 3:
                    continue
                events += 1
                t = act[i, k]
                live = [c for c in cands if t < open_until[c]]
                if live:
                    wrong["equipment_only"] += 1
                    inwin = [c for c in live if plan[c, 0] - win_before <= t <= plan[c, 3] + win_after]
                    if inwin:
                        wrong["equipment_window"] += 1
                        # stage check: a stale FO only expects POD (rank 4)
                        if k == 3:
                            wrong["equipment_window_stage"] += 1
        for k in names:
            res[k]["wrong"].append(wrong[k] / events * 1000)
            res[k]["first_pass"].append((1 - wrong[k] / events) * 100)
        # early events: FO/event handler created after the first event arrives
        is_early = rng.random(n) < p_early
        lag = rng.lognormal(np.log(20), 1.0, n)  # creation lag, median 20 min
        n_e = int(is_early.sum())
        early["discard_lost"].append(n_e / (4 * n) * 1000)
        early["hold_lost"].append(int((is_early & (lag > retry_hold)).sum()) / (4 * n) * 1000)
        early.setdefault("share_recovered", []).append(
            float(np.mean(lag[is_early] <= retry_hold)) * 100 if n_e else 100.0)
    return res, early


def summarize(d):
    if isinstance(d, dict):
        return {k: summarize(v) for k, v in d.items()}
    a = np.array(d, float)
    m = float(np.mean(a))
    h = float(1.96 * np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return [round(m, 3), round(h, 3)]


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else REPS
    out = {"params": P, "n_fo": N_FO, "reps": reps}
    if which in ("all", "e1"):
        out["e1"] = summarize(run_e1(P, reps))
        sens = {}
        for ph in (0.05, 0.20):
            pp = dict(P, p_hold=ph)
            sens[str(ph)] = summarize(run_e1(pp, max(5, reps // 4), 5000))
        out["e1_sens"] = sens
    if which in ("all", "e2"):
        out["e2"] = summarize(run_e2(P, reps))
    if which in ("all", "e3"):
        a, b = run_e3(P, reps)
        out["e3"] = summarize(a)
        out["e3_fan"] = summarize(b)
    if which in ("all", "e4"):
        a, b = run_e4(P, reps)
        out["e4"] = summarize(a)
        out["e4_early"] = summarize(b)
    print(json.dumps(out, indent=1))
