"""모델 가중치(BatchNorm 접어서)를 web/gomoku.html 안의 base64로 넣기
사용: .venv/bin/python export_web.py model15_r7.pt [--size 15 --blocks 3]"""
import argparse, base64, re
import numpy as np, torch
from gomoku import Net


def fold(seq):
    conv, bn = seq[0], seq[1]
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return conv.weight * s[:, None, None, None], (conv.bias - bn.running_mean) * s + bn.bias


p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("--size", type=int, default=15)
p.add_argument("--blocks", type=int, default=3)
p.add_argument("--out", default="web/gomoku.html")
a = p.parse_args()

net = Net(a.size, a.blocks)
net.load_state_dict(torch.load(a.model))
net.eval()
ts = list(fold(net.body[0]))
for blk in net.body[2:]:
    ts += [*fold(blk.a), *fold(blk.b)]
ts += [net.pol.weight, net.pol.bias, *fold(net.val[0]), net.val[3].weight, net.val[3].bias, net.val[5].weight, net.val[5].bias]
w = np.concatenate([t.detach().numpy().ravel() for t in ts]).astype("<f4")
cfg = f'const CFG = {{n: {a.size}, blocks: {a.blocks}, ch: {net.body[0][0].out_channels}}};'
b64 = base64.b64encode(w.tobytes()).decode()

html = open(a.out, encoding="utf-8").read()
html = re.sub(r"^const CFG = .*$", cfg, html, count=1, flags=re.M)
html = re.sub(r'^const W64 = ".*";$', lambda _: f'const W64 = "{b64}";', html, count=1, flags=re.M)
open(a.out, "w", encoding="utf-8").write(html)
print(f"{a.out}: {w.size} weights, {len(b64) / 1e6:.2f} MB base64")
