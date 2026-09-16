"""District tele-screening capacity model (SimPy prototype of the planned SimEvents model).

Answers one question: the minimum number of ophthalmologists needed so that a target
share of patients get a validated report within 48 hours, for a given screening policy.

Flow per patient:
  arrive at a camp -> capture (paused by power outages) -> quality gate
  -> [ungradable: in-camp recapture, else referred for in-person exam]
  -> consent (ABDM OTP, may fail and retry) -> edge inference -> upload (outage-aware)
  -> review queue (ophthalmologists, limited hours) -> report
Policies:
  manual  every gradable image graded by an ophthalmologist (no AI)
  certus  AI grades all; humans review AI-referable + refuse-to-guess cases + a QA sample
          of AI-negatives, each with the 30-second Certus report
Parameters and sources: sim/parameters.md
"""
import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np
import simpy

MIN, HOUR, DAY = 1.0, 60.0, 1440.0


@dataclass
class Params:
    annual_patients: int = 100_000
    camps: int = 20                       # camera sites operating in parallel
    camp_days_per_week: int = 5
    camp_hours: float = 6.0               # 09:00-15:00
    capture_min: float = 5.0              # mean, lognormal
    ungradable_raw: float = 0.225         # SMART India field rate without quality handling
    recapture_success: float = 0.5        # an immediate re-take fixes half of those
    certus_recovery: float = 0.5          # share of borderline images enhancement makes gradable (assumption)
    referable_prev: float = 0.08
    sens: float = 0.92                    # AI operating point (to be replaced by measured numbers)
    spec: float = 0.88
    abstain: float = 0.10                 # refuse-to-guess share (conformal)
    qa_negatives: float = 0.05            # random QA of AI-negatives
    consent_min: float = 2.0
    consent_fail: float = 0.05
    inference_s: float = 3.0
    upload_min: float = 0.5               # report packet (edge inference)
    outages_per_day: float = 1.5
    outage_hours_per_day: float = 1.4     # 24 - 22.6 h average supply
    review_manual_s: float = 90.0
    review_certus_s: float = 30.0
    reviewer_hours_per_day: float = 4.0
    reviewer_days_per_week: int = 5
    tat_target_h: float = 48.0
    days: int = 56                        # simulated horizon (after 7-day warm-up)
    warmup_days: int = 7


def _weekday(t):
    return int(t // DAY) % 7


class District:
    def __init__(self, env, p, policy, reviewers, rng):
        self.env, self.p, self.policy, self.rng = env, p, policy, rng
        self.review = simpy.PriorityResource(env, capacity=reviewers)
        self.power_ok = [True] * p.camps
        self.power_evt = [env.event() for _ in range(p.camps)]
        self.records = []
        self.busy_min = 0.0
        self.referred_in_person = 0
        for c in range(p.camps):
            env.process(self.power(c))
            env.process(self.camp(c))

    # ------------------------------------------------------------ infrastructure
    def power(self, c):
        p, rng = self.p, self.rng
        mean_up = (24 - p.outage_hours_per_day) / p.outages_per_day * HOUR
        mean_down = p.outage_hours_per_day / p.outages_per_day * HOUR
        while True:
            yield self.env.timeout(rng.exponential(mean_up))
            self.power_ok[c] = False
            yield self.env.timeout(rng.exponential(mean_down))
            self.power_ok[c] = True
            self.power_evt[c].succeed()
            self.power_evt[c] = self.env.event()

    def wait_power(self, c):
        while not self.power_ok[c]:
            yield self.power_evt[c]

    def open_hours(self, t):
        return _weekday(t) < self.p.camp_days_per_week and 9 * HOUR <= t % DAY < (9 + self.p.camp_hours) * HOUR

    def reviewer_on(self, t):
        return _weekday(t) < self.p.reviewer_days_per_week and 10 * HOUR <= t % DAY < (10 + self.p.reviewer_hours_per_day) * HOUR

    def until_reviewer_on(self, t):
        d = 0.0
        while not self.reviewer_on(t + d):
            d += 5 * MIN
        return d

    # ------------------------------------------------------------ patients
    def camp(self, c):
        p, rng = self.p, self.rng
        per_camp_day = p.annual_patients / (p.camps * 52 * p.camp_days_per_week)
        rate = per_camp_day / (p.camp_hours * HOUR)       # arrivals per minute while open
        while True:
            t = self.env.now
            if not self.open_hours(t):
                yield self.env.timeout(5 * MIN)
                continue
            yield self.env.timeout(rng.exponential(1 / rate))
            if self.open_hours(self.env.now):
                self.env.process(self.patient(c))

    def patient(self, c):
        p, rng, env = self.p, self.rng, self.env
        t0 = env.now
        yield from self.wait_power(c)
        yield env.timeout(rng.lognormal(np.log(p.capture_min), 0.35))
        ung = p.ungradable_raw * ((1 - p.certus_recovery) if self.policy == "certus" else 1.0)
        if rng.random() < ung:
            yield env.timeout(rng.lognormal(np.log(p.capture_min), 0.35))      # immediate re-take
            if rng.random() > p.recapture_success:
                self.referred_in_person += 1
                self.records.append((t0, np.nan, "ungradable"))
                return
        while True:                                                            # consent
            yield env.timeout(rng.lognormal(np.log(p.consent_min), 0.5))
            if rng.random() > p.consent_fail:
                break
        if self.policy == "certus":
            yield env.timeout(p.inference_s / 60)
        yield from self.wait_power(c)
        yield env.timeout(rng.exponential(p.upload_min))

        referable = rng.random() < p.referable_prev
        if self.policy == "manual":
            needs, prio, dur = True, 1, p.review_manual_s
        else:
            if rng.random() < p.abstain:
                needs, prio = True, 0
            else:
                pos = rng.random() < (p.sens if referable else 1 - p.spec)
                needs, prio = (True, 0) if pos else (rng.random() < p.qa_negatives, 2)
            dur = p.review_certus_s
        if not needs:
            self.records.append((t0, env.now, "auto"))
            return
        with self.review.request(priority=prio) as req:
            yield req
            yield env.timeout(self.until_reviewer_on(env.now))
            d = rng.lognormal(np.log(dur / 60), 0.4)
            self.busy_min += d
            yield env.timeout(d)
        self.records.append((t0, env.now, "reviewed"))


def simulate(p, policy, reviewers, seed):
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    d = District(env, p, policy, reviewers, rng)
    env.run(until=(p.days + p.warmup_days) * DAY)
    rec = [(a, b, k) for a, b, k in d.records if a >= p.warmup_days * DAY]
    done = np.array([(b - a) / HOUR for a, b, k in rec if k != "ungradable"])
    arrived_late = (p.days + p.warmup_days) * DAY - p.tat_target_h * HOUR
    eligible = [(a, b) for a, b, k in rec if k != "ungradable" and a <= arrived_late]
    within = np.mean([(b - a) / HOUR <= p.tat_target_h for a, b in eligible]) if eligible else np.nan
    avail = reviewers * p.days * (p.reviewer_days_per_week / 7) * p.reviewer_hours_per_day * HOUR
    return {"patients": len(rec), "reported": int(len(done)), "within_target": float(within),
            "tat_p50_h": float(np.median(done)) if len(done) else np.nan,
            "tat_p95_h": float(np.percentile(done, 95)) if len(done) else np.nan,
            "reviewer_utilisation": d.busy_min / avail,
            "ungradable_referred": sum(k == "ungradable" for _, _, k in rec) / max(len(rec), 1),
            "human_reviews_share": sum(k == "reviewed" for _, _, k in rec) / max(len(rec), 1)}


def min_reviewers(p, policy, target=0.95, reps=3, n_max=60):
    """Smallest reviewer count whose mean share within 48 h >= target (reps seeds each)."""
    lo, hi, table = 1, n_max, {}

    def ok(n):
        if n not in table:
            runs = [simulate(p, policy, n, seed) for seed in range(reps)]
            table[n] = {k: float(np.nanmean([r[k] for r in runs])) for k in runs[0]}
        return table[n]["within_target"] >= target

    if not ok(hi):
        return None, table
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--out", default="D:/Certus/sim/capacity_results.json")
    a = ap.parse_args()
    p = Params()
    out = {"params": asdict(p), "target_share_within_48h": a.target, "policies": {}}
    for policy in ("manual", "certus"):
        n, table = min_reviewers(p, policy, a.target)
        out["policies"][policy] = {"min_ophthalmologists": n, "at_min": table.get(n), "searched": table}
        print(f"{policy:7s}: minimum ophthalmologists = {n}  ->  {json.dumps(table.get(n), default=float)}")
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
