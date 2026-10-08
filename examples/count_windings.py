"""Count the windings between two points with the winding-count network (CPU, a few seconds).

  python examples/count_windings.py [--weights counter_model.pt] [--samples examples/sample_profiles.npz]

The samples are profiles of 9 point pairs of the held-out part of the W2 slab (14 channels sampled every L2 voxel along the line between the two points: CT, surface prediction,
phase-network output, side lines, Lasagna; see the docstring of src/counter_net.py) with their true winding counts. On the first run the weights (3 MB) are downloaded from
https://huggingface.co/qizhiran/vesuvius-phase-field-winding . The network alone gives a count and a calibrated confidence; the decision to certify a count is made by the
combined certifier (src/certnet.py), which needs ladder row files that are not included.
Needs numpy and torch. The sample profiles are derived from Vesuvius Challenge open data (CC BY-NC 4.0)."""
import os, sys, argparse, urllib.request
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import counter_net as cn

URL = "https://huggingface.co/qizhiran/vesuvius-phase-field-winding/resolve/main/counter_network/model.pt"
ap = argparse.ArgumentParser(); ap.add_argument("--weights", default=os.path.join(os.getcwd(), "counter_model.pt")); ap.add_argument("--samples", default=os.path.join(HERE, "sample_profiles.npz")); a = ap.parse_args()
if not os.path.exists(a.weights):
    print("downloading the weights from", URL, flush=True); urllib.request.urlretrieve(URL, a.weights)
ck = torch.load(a.weights, map_location="cpu", weights_only=True); cfg = ck["config"]
net = cn.CounterNet(s_cols=cfg["s_cols"], s_abs=cfg["s_abs"], c1=cfg["c1"], c2=cfg["c2"], derive=cfg["derive"], dils=cfg["dils"]); net.load_state_dict(ck["state"]); net.eval(); T = float(ck["temperature"])
d = np.load(a.samples); P = torch.from_numpy(d["P"].astype(np.float32)); S = torch.from_numpy(d["S"]); truth = d["k"].astype(int)
with torch.no_grad(): prob = torch.softmax(net(P, S) / T, 1).numpy()
length = (d["P"][:, 13].astype(np.float32).sum(-1)).astype(int)
print(f"{'pair':>4} {'line length (voxels)':>21} {'true windings':>14} {'network count':>14} {'confidence':>11}")
for i in range(len(truth)):
    c = int(prob[i].argmax()); name = "8 or more" if c == 8 else str(c)
    print(f"{i:>4} {length[i]:>21} {truth[i]:>14} {name:>14} {prob[i].max():>11.3f}")
print(f"\nright on {int((prob.argmax(1) == truth).sum())} of {len(truth)} pairs")
