"""How well does the mesh locate its own boards? (3-D simulated halls, never seen in training)"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hallsense.selfmap import self_map, align

errs_sim, errs_1d, halls = [], [], []
for f in sorted(glob.glob("data3/test/*_s0.npz")):
    z = np.load(f)
    P = np.nanmean(z["cal_mean"].astype(np.float32), axis=(1, 2))
    links = z["links"]; true = z["nodes"][:, :2]; n = len(true)
    X, ne = self_map(P, links, n)
    A = align(X, true, scale=True)
    e = np.linalg.norm(A - true, axis=1)
    # realistic: scale from ONE tape-measured distance (boards 0 and 1)
    s = np.linalg.norm(true[0] - true[1]) / np.linalg.norm(X[0] - X[1])
    B = align(X * s, true, scale=False)
    e1 = np.linalg.norm(B - true, axis=1)
    errs_sim += list(e); errs_1d += list(e1)
    halls.append(dict(file=os.path.basename(f), W=float(z["W"]), D=float(z["D"]), median_err=float(np.median(e)), median_err_1tape=float(np.median(e1)), n_exp=float(ne)))
res = dict(halls=len(halls), node_error_median_m=float(np.median(errs_1d)), node_error_p80_m=float(np.percentile(errs_1d, 80)),
           shape_only_median_m=float(np.median(errs_sim)), per_hall=halls)
os.makedirs("results", exist_ok=True)
json.dump(res, open("results/selfmap.json", "w"), indent=1)
print({k: v for k, v in res.items() if k != "per_hall"})

# with the wall prior (boards are on the walls, hall size typed in)
from hallsense.selfmap import self_map_walls
ew = []
for f in sorted(glob.glob("data3/test/*_s0.npz")):
    z = np.load(f); P = np.nanmean(z["cal_mean"].astype(np.float32), axis=(1, 2))
    true = z["nodes"][:, :2]; n = len(true)
    X = self_map_walls(P, z["links"], n, float(z["W"]), float(z["D"]))
    # the 8 symmetries of a rectangle are unknowable from radio alone; the user names the front wall
    best = np.inf
    for fx in (False, True):
        for fy in (False, True):
            Y = X.copy()
            if fx: Y[:, 0] = float(z["W"]) - Y[:, 0]
            if fy: Y[:, 1] = float(z["D"]) - Y[:, 1]
            best = min(best, np.median(np.linalg.norm(Y - true, axis=1)))
    ew.append(best)
res["walls_prior_median_m"] = float(np.median(ew)); res["walls_prior_per_hall"] = [float(x) for x in ew]
json.dump(res, open("results/selfmap.json", "w"), indent=1)
print("with wall prior: median over halls of median node error", np.median(ew).round(2), "m;", np.round(ew, 1))
