import numpy as np
import track_trace_simulation as sim

def test_orders_are_strictly_ordered():
      rng = np.random.default_rng(1)
      plan, act = sim.gen_orders(rng, sim.P, 200)
      assert plan.shape == (200, 4)
      assert act.shape == (200, 4)
      assert np.all(np.diff(act, axis=1) > 0)

def test_receipts_never_precede_events():
      rng = np.random.default_rng(2)
      plan, act = sim.gen_orders(rng, sim.P, 200)
      rec = sim.gen_receipts(rng, sim.P, act)
      assert np.all(rec >= act)

def test_atomic_duplicate_check_leaves_no_duplicates():
      res, fan = sim.run_e3(sim.P, reps=2, n_events=2000)
      assert all(v == 0.0 for v in res["atomic"])
      assert all(n <= d for n, d in zip(res["nonatomic"], res["none"]))
      assert all(v == 0.0 for v in fan["per_target"]["dup"])

def test_summarize_returns_mean_and_halfwidth():
      out = sim.summarize({"a": [1.0, 3.0]})
      assert out["a"][0] == 2.0
  
