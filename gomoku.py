"""렌주 룰 오목 AlphaZero (자기대국 MCTS + ResNet)
  train  : 자기대국 학습            duel   : 두 모델 대국 승률
  ladder : 1회 학습마다 직전 모델과 대국, 향상이 멈추면 종료
  play   : CLI 대국                  test   : 규칙 self-check"""
import argparse, math, os, queue, random, shutil, time
from collections import deque
import multiprocessing as mp
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))
MAX_DEPTH = 2  # 3-3 판정 시 완성점 금수 여부를 재귀 확인하는 깊이


# ---------- 렌주 규칙 (흑: 3-3, 4-4, 장목 금지 / 정확히 5목만 승리, 백: 5목 이상 승리) ----------

def run(g, r, c, dr, dc):
    """(r,c)를 지나는 같은 색 연속 구간 [lo, hi] (방향 오프셋)"""
    n, p, lo, hi = len(g), g[r, c], 0, 0
    while 0 <= r + (lo - 1) * dr < n and 0 <= c + (lo - 1) * dc < n and g[r + (lo - 1) * dr, c + (lo - 1) * dc] == p:
        lo -= 1
    while 0 <= r + (hi + 1) * dr < n and 0 <= c + (hi + 1) * dc < n and g[r + (hi + 1) * dr, c + (hi + 1) * dc] == p:
        hi += 1
    return lo, hi


def empties_on_line(g, r, c, dr, dc):
    n = len(g)
    for k in range(-4, 5):
        y, x = r + k * dr, c + k * dc
        if k and 0 <= y < n and 0 <= x < n and g[y, x] == 0:
            yield k, y, x


def fours(g, r, c, dr, dc):
    """흑 (r,c)를 포함해 한 수로 정확히 5목이 되는 4의 개수 (열린 4는 1개)"""
    ks = []
    for k, y, x in empties_on_line(g, r, c, dr, dc):
        g[y, x] = 1
        lo, hi = run(g, y, x, dr, dc)
        g[y, x] = 0
        if hi - lo == 4 and k + lo <= 0 <= k + hi:
            ks.append(k)
    return 1 if len(ks) == 2 and ks[1] - ks[0] == 5 else len(ks)


def open_three(g, r, c, dr, dc, depth):
    """한 수(금수 아님)로 열린 4(_XXXX_, 양끝 모두 정확히 5)가 되면 열린 3"""
    n = len(g)
    inb = lambda y, x: 0 <= y < n and 0 <= x < n
    for k, y, x in empties_on_line(g, r, c, dr, dc):
        g[y, x] = 1
        lo, hi = run(g, y, x, dr, dc)
        ok = hi - lo == 3 and k + lo <= 0 <= k + hi
        for e in (lo - 1, hi + 1):
            ey, ex = y + e * dr, x + e * dc
            by, bx = ey + (-dr if e == lo - 1 else dr), ex + (-dc if e == lo - 1 else dc)  # 끝 바깥칸: 흑이면 장목
            ok = ok and inb(ey, ex) and g[ey, ex] == 0 and not (inb(by, bx) and g[by, bx] == 1)
        g[y, x] = 0
        if ok and (depth >= MAX_DEPTH or not forbidden(g, y, x, depth + 1)):
            return True
    return False


def forbidden(g, r, c, depth=0):
    """흑이 빈 칸 (r,c)에 두는 것이 금수인지. 정확히 5목을 만들면 금수 아님"""
    g[r, c] = 1
    try:
        lens = [hi - lo + 1 for lo, hi in (run(g, r, c, *d) for d in DIRS)]
        if 5 in lens:
            return False
        if max(lens) > 5:
            return True
        n4 = n3 = 0
        for d in DIRS:
            f = fours(g, r, c, *d)
            n4 += f
            if not f and open_three(g, r, c, *d, depth):
                n3 += 1
        return n4 >= 2 or n3 >= 2
    finally:
        g[r, c] = 0


def candidates(g):
    """금수 가능성이 있는 빈 칸만 추림: 두 방향 이상에 ±4 내 흑 2개+, 또는 한 방향 ±5 내 흑 5개+"""
    n, B = len(g), np.pad(g == 1, 5).astype(np.int8)
    two, five = np.zeros((n, n), np.int8), np.zeros((n, n), bool)
    for dr, dc in DIRS:
        sh = lambda k: B[5 + k * dr:5 + k * dr + n, 5 + k * dc:5 + k * dc + n]
        two += sum(sh(k) for k in (-4, -3, -2, -1, 1, 2, 3, 4)) >= 2
        five |= sum(sh(k) for k in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)) >= 5
    return ((two >= 2) | five) & (g == 0)


class Board:
    def __init__(self, n=15):
        self.n, self.grid, self.player, self.last = n, np.zeros((n, n), np.int8), 1, None

    def copy(self):
        b = Board.__new__(Board)
        b.n, b.grid, b.player, b.last = self.n, self.grid.copy(), self.player, self.last
        return b

    def legal(self):
        g = self.grid
        empty = g.ravel() == 0
        if self.player == 1:
            for m in np.flatnonzero(candidates(g)):
                if forbidden(g, *divmod(m, self.n)):
                    empty[m] = False
        return np.flatnonzero(empty)

    def play(self, m):
        self.grid.flat[m] = self.player
        self.player, self.last = -self.player, m

    def winner(self):
        """None: 진행 중, 0: 무승부(꽉 참), 1/-1: 승자"""
        if self.last is None:
            return None
        r, c = divmod(self.last, self.n)
        p = int(self.grid[r, c])
        lens = [hi - lo + 1 for lo, hi in (run(self.grid, r, c, *d) for d in DIRS)]
        if (5 in lens) if p == 1 else (max(lens) >= 5):
            return p
        return 0 if not (self.grid == 0).any() else None

    def result(self):
        """winner()에 '흑이 둘 곳이 금수뿐'인 무승부까지 포함 (게임 루프용)"""
        w = self.winner()
        return 0 if w is None and not len(self.legal()) else w

    def state(self):
        g = self.grid
        return np.stack([g == self.player, g == -self.player, np.full_like(g, self.player == 1, bool)]).astype(np.float32)


# ---------- 신경망 ----------

def conv_bn(i, o, k=3):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2), nn.BatchNorm2d(o))


class Block(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.a, self.b = conv_bn(ch, ch), conv_bn(ch, ch)

    def forward(self, x):
        return F.relu(x + self.b(F.relu(self.a(x))))


class Net(nn.Module):
    """conv 1층 + 잔차 블록(conv 2층씩). 수용영역 = 3 + 4*blocks >= 보드 크기"""

    def __init__(self, n=15, blocks=3, ch=64):
        super().__init__()
        self.body = nn.Sequential(conv_bn(3, ch), nn.ReLU(), *[Block(ch) for _ in range(blocks)])
        self.pol = nn.Conv2d(ch, 1, 1)
        self.val = nn.Sequential(conv_bn(ch, 1, 1), nn.ReLU(), nn.Flatten(), nn.Linear(n * n, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

    def forward(self, x):
        h = self.body(x)
        return self.pol(h).flatten(1), self.val(h).squeeze(1)


def load_net(a, path=None):
    net = Net(a.size, a.blocks)
    net.load_state_dict(torch.load(path or a.model))
    return net.eval()


# ---------- MCTS ----------

class Node:
    __slots__ = ("P", "N", "W", "kids")

    def __init__(self, p):
        self.P, self.N, self.W, self.kids = p, 0, 0.0, {}


@torch.no_grad()
def evaluate(net, b):
    legal = b.legal()
    if not len(legal):
        return legal, legal, 0.0
    logits, v = net(torch.from_numpy(b.state())[None])
    return legal, torch.softmax(logits[0, legal], 0).numpy(), v.item()


def mcts(b, net, sims, noise=False, c=1.5):
    """루트의 수별 방문 비율 π (규칙 지름길 없이 탐색만으로 결정)"""
    pi = np.zeros(b.n * b.n, np.float32)
    legal, p, _ = evaluate(net, b)
    if noise:
        p = 0.75 * p + 0.25 * np.random.dirichlet([0.3] * len(p))
    root = Node(1.0)
    root.kids = {m: Node(q) for m, q in zip(legal, p)}
    for _ in range(sims):
        node, bb, path = root, b.copy(), [root]
        while node.kids:
            sq, fpu = math.sqrt(node.N), (-node.W / node.N if node.N else 0)  # 안 가본 수 = 현재 국면 평가값
            m, node = max(node.kids.items(), key=lambda kv: (kv[1].W / kv[1].N if kv[1].N else fpu) + c * kv[1].P * sq / (1 + kv[1].N))
            bb.play(m)
            path.append(node)
        w = bb.winner()
        if w is None:
            legal, p, v = evaluate(net, bb)
            node.kids = {m: Node(q) for m, q in zip(legal, p)}
        else:
            v = -1.0 if w else 0.0  # 직전 수를 둔 쪽이 이김 → 둘 차례인 쪽 기준 -1
        for nd in reversed(path):
            v = -v
            nd.N += 1
            nd.W += v
    for m, nd in root.kids.items():
        pi[m] = nd.N
    return pi / max(pi.sum(), 1)


# ---------- 학습 ----------

def self_play(args):
    sd, n, sims, blocks = args
    torch.set_num_threads(1)
    net = Net(n, blocks)
    net.load_state_dict(sd)
    net.eval()
    b, hist = Board(n), []
    while (w := b.result()) is None:
        pi = mcts(b, net, sims, noise=True)
        hist.append((b.state(), pi, b.player))
        b.play(np.random.choice(len(pi), p=pi) if len(hist) <= 8 else int(pi.argmax()))
    return [(s, pi, float(w * pl)) for s, pi, pl in hist]


def augment(s, pi, z, n):
    pi = pi.reshape(n, n)
    for k in range(4):
        for f in (False, True):
            ss, pp = np.rot90(s, k, (1, 2)), np.rot90(pi, k)
            if f:
                ss, pp = ss[:, :, ::-1], pp[:, ::-1]
            yield ss.copy(), pp.ravel().copy(), z


def train(a):
    net = load_net(a) if os.path.exists(a.model) else Net(a.size, a.blocks).eval()
    opt = torch.optim.Adam(net.parameters(), a.lr, weight_decay=1e-4)
    buf, done, sent, q = deque(maxlen=100_000), 0, 0, queue.Queue()
    torch.set_num_threads(2)  # 나머지 코어는 자기대국 워커용

    def submit():
        nonlocal sent
        sent += 1
        sd = {k: v.clone() for k, v in net.state_dict().items()}
        pool.apply_async(self_play, ((sd, a.size, a.sims, a.blocks),), callback=q.put, error_callback=q.put)

    # 끝난 게임 자리에 바로 새 게임 투입, workers판 모일 때마다 학습 (진행 중 게임은 이전 가중치 사용)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for _ in range(min(a.workers, a.games)):
            submit()
        while done < a.games:
            t, k, games = time.time(), min(a.workers, a.games - done), []
            while len(games) < k:
                g = q.get()
                if isinstance(g, Exception):
                    raise g
                games.append(g)
                if sent < a.games:
                    submit()
            done += k
            for g in games:
                for s, pi, z in g:
                    buf.extend(augment(s, pi, z, a.size))
            net.train()
            for _ in range(a.steps):
                s, pi, z = map(np.array, zip(*random.sample(buf, min(256, len(buf)))))
                logits, v = net(torch.from_numpy(s))
                loss = F.mse_loss(v, torch.from_numpy(z).float()) - (torch.from_numpy(pi) * F.log_softmax(logits, 1)).sum(1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            torch.save(net.state_dict(), a.model)
            wmax = np.mean([pi.max() for g in games for s, pi, z in g if s[2, 0, 0] == 0])  # 백 최대 방문 비율, ~0이면 붕괴
            print(f"games {done}/{a.games}  loss {loss.item():.3f}  avg_len {sum(map(len, games)) / k:.1f}  white_max {wmax:.2f}  {time.time() - t:.0f}s", flush=True)
            if a.ckpt and done // a.ckpt > (done - k) // a.ckpt:
                n = done // a.ckpt * a.ckpt
                tag = f"{n // 1000}k" if n % 1000 == 0 else str(n)
                torch.save(net.state_dict(), a.model.replace(".pt", f"_{tag}.pt"))
                print(f"  checkpoint {tag}  열린3 막기(가로, 대각): {tactics(net, a.size, a.sims)}", flush=True)


def tactics(net, n, sims):
    th = torch.get_num_threads()
    torch.set_num_threads(1)
    c, res = n // 2, ""
    for line, want in (([(c, c - 1), (c, c), (c, c + 1)], [(c, c - 2), (c, c + 2), (c, c - 3), (c, c + 3)]),
                       ([(c - 1, c - 1), (c, c), (c + 1, c + 1)], [(c - 2, c - 2), (c + 2, c + 2), (c - 3, c - 3), (c + 3, c + 3)])):
        b = Board(n)
        for r, cc in line:
            b.grid[r, cc] = 1
        b.grid[0, 0] = b.grid[n - 1, n - 1] = -1
        b.player = -1
        res += "O" if int(mcts(b, net, sims).argmax()) in [r * n + cc for r, cc in want] else "X"
    torch.set_num_threads(th)
    return res


# ---------- 평가 / 대국 ----------

def duel_game(args):
    pa, pb, i, a = args
    torch.set_num_threads(1)
    np.random.seed(i)
    A, B = load_net(a, pa), load_net(a, pb)
    nets = {1: A, -1: B} if i % 2 == 0 else {1: B, -1: A}
    b = Board(a.size)
    while (w := b.result()) is None:
        pi = mcts(b, nets[b.player], a.sims)
        b.play(np.random.choice(len(pi), p=pi) if (b.grid != 0).sum() < 4 else int(pi.argmax()))  # 초반 4수 무작위로 다양화
    return 0.5 if w == 0 else float(nets[w] is A), i % 2 == 0


def duel(a, pa=None, pb=None):
    """A의 승률 (흑/백 반반, 무승부 0.5)"""
    pa, pb = pa or a.model, pb or a.other
    with mp.get_context("spawn").Pool(min(a.duel_games, a.workers)) as pool:
        r = pool.map(duel_game, [(pa, pb, i, a) for i in range(a.duel_games)])
    rate = sum(x for x, _ in r) / len(r)
    blk, wht = sum(x for x, k in r if k), sum(x for x, k in r if not k)
    print(f"{pa} vs {pb}: 흑 {blk:.1f}/{len(r) // 2}  백 {wht:.1f}/{len(r) - len(r) // 2}  승률 {rate:.3f}", flush=True)
    return rate


def ladder(a):
    stem, low = a.model.replace(".pt", ""), 0
    shutil.copy(a.model, f"{stem}_r0.pt")
    a.ckpt = 0
    for i in range(1, a.rounds + 1):
        train(a)
        shutil.copy(a.model, f"{stem}_r{i}.pt")
        print(f"round {i}: ", end="", flush=True)
        low = low + 1 if duel(a, f"{stem}_r{i}.pt", f"{stem}_r{i - 1}.pt") < 0.55 else 0
        if low >= 2:
            print(f"STOP: 2회 연속 55% 미만 (round {i})", flush=True)
            return
    print("STOP: 최대 회차 도달", flush=True)


def show(b):
    print("   " + " ".join(f"{i:x}" for i in range(b.n)))
    for r in range(b.n):
        print(f"{r:2x} " + " ".join(".XO"[v] for v in b.grid[r]))


def play(a):
    torch.set_num_threads(1)
    net, b, human = load_net(a), Board(a.size), 1 if a.first else -1
    while (w := b.result()) is None:
        show(b)
        if b.player == human:
            try:
                r, c = (int(t, 16) for t in input("행 열 (16진수)> ").split())
                m = r * b.n + c
                assert 0 <= r < b.n and 0 <= c < b.n and m in b.legal()
            except (ValueError, AssertionError):
                print("둘 수 없는 자리 (형식 오류, 이미 돌 있음, 또는 흑 금수)")
                continue
        else:
            m = int(mcts(b, net, a.sims).argmax())
            print(f"봇: {m // b.n:x} {m % b.n:x}")
        b.play(m)
    show(b)
    print("무승부" if w == 0 else "승리!" if w == human else "패배")


def test(a):
    def board(black=(), white=()):
        b = Board()
        for r, c in black:
            b.grid[r, c] = 1
        for r, c in white:
            b.grid[r, c] = -1
        return b

    fb = lambda b, r, c: forbidden(b.grid, r, c)
    assert fb(board([(7, 5), (7, 6), (5, 7), (6, 7)]), 7, 7), "3-3"
    assert not fb(board([(7, 5), (7, 6), (5, 7), (6, 7)], [(7, 4)]), 7, 7), "한쪽 막힌 3은 3-3 아님"
    assert fb(board([(7, 4), (7, 6), (5, 7), (6, 7)]), 7, 7), "띈 3(X_XX) 포함 3-3"
    assert fb(board([(7, 4), (7, 5), (7, 6), (4, 7), (5, 7), (6, 7)]), 7, 7), "4-4"
    assert fb(board([(7, 3), (7, 5), (7, 7), (7, 9)]), 7, 6), "한 줄 4-4 (X_XXX_X)"
    assert fb(board([(7, 1), (7, 2), (7, 3), (7, 4), (7, 6)]), 7, 5), "장목"
    assert not fb(board([(7, 3), (7, 4), (7, 5), (7, 6), (5, 7), (6, 7)]), 7, 7), "5목+3-3 은 승리"
    assert not fb(board([(7, 4), (7, 5), (7, 6), (5, 7), (6, 7)]), 7, 7), "4-3 허용"
    b = board(white=[(7, 1), (7, 2), (7, 3), (7, 4), (7, 6)])
    b.player = -1
    b.play(7 * 15 + 5)
    assert b.winner() == -1, "백 장목 승리"
    b = board([(7, 1), (7, 2), (7, 3), (7, 4), (7, 6)])
    b.play(7 * 15 + 5)
    assert b.winner() is None, "흑 장목은 승리 아님"
    b = board([(7, 5), (7, 6), (5, 7), (6, 7)])
    assert 7 * 15 + 7 not in b.legal(), "금수는 legal 제외"
    pi = mcts(board([(7, 5), (7, 6), (5, 7), (6, 7)], [(0, 0), (14, 14), (0, 14), (14, 0)]), Net().eval(), 50)
    assert abs(pi.sum() - 1) < 1e-5 and pi[7 * 15 + 7] == 0, "MCTS는 금수에 방문 안 함"
    print("ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "ladder", "duel", "play", "test"])
    p.add_argument("--model", default="renju.pt")
    p.add_argument("--other", help="duel 상대 모델")
    p.add_argument("--size", type=int, default=15)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--sims", type=int, default=400)
    p.add_argument("--games", type=int, default=10000, help="ladder에서는 회차당 판 수")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--ckpt", type=int, default=1000, help="N판마다 체크포인트 저장 (0: 안 함)")
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--duel-games", type=int, default=80)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    p.add_argument("--first", action="store_true", help="play: 사람이 흑")
    a = p.parse_args()
    globals()[a.cmd](a)
