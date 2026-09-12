"""nn_layout.py：用神经网络（代理模型 + 搜索）寻找最优测量点布局。

思路（论文可用的"数据驱动布局寻优"方法）：
1. **代理模型**：多层感知机 f(布局) → 预测"听到率"（定向半圆盘命中 + 全向圆盘命中两个头）。
   训练数据 = 大量随机 / 结构启发布局 + **向量化蒙特卡洛**听到率标签（每布局 2 万算例）。
2. **搜索**：以代理预测为评分，多起点局部搜索（高斯扰动爬山 + 重启）在连续布局空间中找
   预测听到率最高（漏听率最低）的布局。
3. **验证**：对代理输出的候选布局用高精度蒙特卡洛（40 万）与贴边对抗复核，再算 TSP 访问
   路程 / 20 局演练，与手工优化布局对比——**代理只是近似，最终结论一律以精确验证为准**。

运行（示例）：
    python -X utf8 -m cumcm.t4.nn_layout --gen 4000 --mc-n 20000 --train-epochs 80 \
        --search-starts 8 --search-iters 4000 --validate-n 400000
"""
from __future__ import annotations

import argparse
import math
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from cumcm.common.routing import (dist_matrix, nearest_order, two_opt_first,
                                  two_opt_greedy)

MAX_R = 2270.0               # 自由点允许的最大半径（原布局外推点最远 2270）
MC_N_DEFAULT = 20_000        # 每条训练样本的蒙特卡洛算例数
DATA_PATH = "results/t4/.nn_layout_data.npz"
BEST_PATH = "results/t4/.nn_best_layout.npy"


# --------------------------------------------------------------------------
# 1. 向量化蒙特卡洛听到率评估（打标签用）
# --------------------------------------------------------------------------
def fast_miss(pts: Sequence[Sequence[float]], n: int = MC_N_DEFAULT,
              seed: int = 0) -> Tuple[float, float]:
    """向量化评估一个布局的缺听率，返回 (定向缺听, 全向缺听)。

    定向源：存在测量点 p 满足 |p−g| ≤ R 且 (p−g)·u(θ) ≥ 0（半圆盘命中）；
    全向源：存在测量点 p 满足 |p−g| ≤ R（圆盘命中）。场景：|g| 面积均匀到 1770、
    θ 均匀、R ∈ [1000, 1500]。与 `sweep.verify_hearing_stats` 的蒙特卡洛口径一致。
    """
    P = np.asarray(pts, dtype=float)
    rng = np.random.default_rng(seed)
    g_r = 1770.0 * np.sqrt(rng.random(n))
    g_a = rng.random(n) * 2.0 * math.pi
    G = np.stack((g_r * np.cos(g_a), g_r * np.sin(g_a)), axis=1)
    th = rng.random(n) * 2.0 * math.pi
    U = np.stack((np.cos(th), np.sin(th)), axis=1)
    Rv = rng.uniform(1000.0, 1500.0, n)
    D = G[:, None, :] - P[None, :, :]                 # (n, N, 2)
    dist2 = np.einsum("ijk,ijk->ij", D, D)
    in_rad = dist2 <= Rv[:, None] * Rv[:, None] + 1e-9        # (n, N)
    proj = np.einsum("ijk,ik->ij", D, U)                       # (n, N)
    dir_hit = in_rad & (proj >= -1e-9)
    heard = dir_hit.any(axis=1)
    omni_heard = in_rad.any(axis=1)
    return 1.0 - heard.mean(), 1.0 - omni_heard.mean()


# --------------------------------------------------------------------------
# 2. 训练数据生成（随机 + 结构启发，覆盖"好布局"区域）
# --------------------------------------------------------------------------
def gen_layouts(n: int, n_free: int, seed: int = 1) -> np.ndarray:
    """生成 n 个布局，每个含 n_free 个自由点（原点固定、不入参）。

    四种来源混合（保证既有多样性、又覆盖"环+内部点"这类好布局）：
      * 40%：含内部点的环形结构（环半径 1400~1950 + 内部点半径 250~950、方位抖动）；
      * 30%：纯均匀随机（半径 ≤ MAX_R，多样性）；
      * 20%：聚合辐条（角向若干径向层，模拟"角扫描"型）；
      * 10%：对已有布局做随机位移（加密布局空间的好邻域）。
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n, n_free, 2), dtype=float)

    def _fill_ring(i: int) -> None:
        n_ring = int(rng.integers(8, 15))
        n_in = n_free - n_ring
        r_ring = rng.uniform(1400.0, 1950.0)
        aa = 2.0 * math.pi * np.arange(n_ring) / n_ring + rng.uniform(-0.12, 0.12)
        rr = np.full(n_ring, r_ring)
        if n_in > 0:
            rr = np.concatenate([rr, rng.uniform(250.0, 950.0, n_in)])
            aa = np.concatenate([aa, rng.uniform(0.0, 2.0 * math.pi, n_in)])
        out[i, :, 0] = rr * np.cos(aa)
        out[i, :, 1] = rr * np.sin(aa)

    for i in range(n):
        r = rng.random()
        if r < 0.40:
            _fill_ring(i)
        elif r < 0.70:
            rr = MAX_R * np.sqrt(rng.random(n_free))
            aa = rng.uniform(0.0, 2.0 * math.pi, n_free)
            out[i, :, 0] = rr * np.cos(aa)
            out[i, :, 1] = rr * np.sin(aa)
        elif r < 0.90:
            n_spokes = int(rng.integers(3, 8))
            layers = rng.integers(1, 4, n_spokes)
            aa = 2.0 * math.pi * np.arange(n_spokes) / n_spokes
            pts = []
            for s in range(n_spokes):
                for _ in range(int(layers[s % n_spokes])):
                    pts.append((rng.uniform(500.0, 2000.0), aa[s]))
            if len(pts) > n_free:
                pts = pts[:n_free]
            while len(pts) < n_free:
                pts.append((rng.uniform(100.0, 2000.0),
                            rng.uniform(0.0, 2.0 * math.pi)))
            rr = np.array([p[0] for p in pts])
            aa = np.array([p[1] for p in pts])
            out[i, :, 0] = rr * np.cos(aa)
            out[i, :, 1] = rr * np.sin(aa)
        else:
            j = int(rng.integers(0, i)) if i else 0
            out[i] = out[j] + rng.normal(0.0, 90.0, out[j].shape)
    return out


def label_layouts(points: np.ndarray, mc_n: int, seed: int = 7
                  ) -> np.ndarray:
    """为 (M, n_free, 2) 的布局批量打标签：返回 (M, 2) = [定向缺听, 全向缺听]。"""
    M = len(points)
    lab = np.empty((M, 2), dtype=float)
    for i in range(M):
        lab[i] = fast_miss(points[i], n=mc_n, seed=seed + i * 7)
    return lab


# --------------------------------------------------------------------------
# 3. 多层感知机（PyTorch）
# --------------------------------------------------------------------------
class Net(nn.Module):
    """两层 MLP：输入 (n_free*2,)（归一化坐标）→ 输出 (2,) = 定向/全向听到率。"""

    def __init__(self, n_in: int, h1: int = 128, h2: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def featurize(layout: np.ndarray) -> np.ndarray:
    """(n_free, 2) 布局 → (n_free*2,) 特征（已 /MAX_R 归一）。"""
    return (layout.reshape(-1) / MAX_R).astype(np.float32)


def proxy_score(net: nn.Module, layout: np.ndarray, w_omni: float = 0.5,
                dev: str = "cpu") -> float:
    """代理评分：越小越好 = 定向与全向缺听的加权和（听率换算）。"""
    x = torch.from_numpy(featurize(layout)[None, :]).to(dev)
    with torch.no_grad():
        y = net(x)[0].cpu().numpy()
    # 输出 = log1p(缺听率) ∈ [0, 0.69]；缺听率 = expm1(y)，clip 下限 0 防负
    d, o = float(np.clip(y[0], 0.0, 8.0)), float(np.clip(y[1], 0.0, 8.0))
    return math.expm1(d) + w_omni * math.expm1(o)


def train_net(X: np.ndarray, Y: np.ndarray, epochs: int = 80,
              batch: int = 256, lr: float = 1.5e-3,
              dev: str = "cpu") -> Tuple[nn.Module, List[float]]:
    """Adam 训练，返回 (网络, 每 epoch 平均损失)。"""
    n = X.shape[0]
    net = Net(X.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    crit = nn.MSELoss()
    Xt = torch.from_numpy(X).to(dev)
    Yt = torch.from_numpy(Y.astype(np.float32)).to(dev)
    losses = []
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        cnt = 0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            loss = crit(net(Xt[idx]), Yt[idx])
            loss.backward()
            opt.step()
            tot += float(loss.item())
            cnt += 1
        losses.append(tot / max(1, cnt))
    return net, losses


# --------------------------------------------------------------------------
# 4. 搜索：多起点局部搜索（高斯扰动爬山）
# --------------------------------------------------------------------------
def search(net: nn.Module, n_free: int, starts: Sequence[np.ndarray],
           iters: int = 4000, sigma: float = 120.0, w_omni: float = 0.5,
           seed: int = 3, dev: str = "cpu") -> Tuple[np.ndarray, float]:
    """从每个起点做高斯扰动爬山：每次随机扰动一个点，代理评分变好才接受。"""
    rng = np.random.default_rng(seed)
    best_layout: Optional[np.ndarray] = None
    best_score = float("inf")
    feats = np.empty((1, n_free * 2), dtype=np.float32)
    for si, start in enumerate(starts):
        cur = start.copy()
        sc = proxy_score(net, cur, w_omni, dev)
        for _ in range(iters):
            j = rng.integers(0, n_free)
            old = cur[j].copy()
            cur[j] += rng.normal(0.0, sigma, 2)
            if np.hypot(cur[j, 0], cur[j, 1]) > MAX_R:
                cur[j] = old
                continue
            feats[0] = featurize(cur)
            xt = torch.from_numpy(feats).to(dev)
            with torch.no_grad():
                y = net(xt)[0].cpu().numpy()
            d, o = float(np.clip(y[0], 0.0, 8.0)), float(np.clip(y[1], 0.0, 8.0))
            sc2 = math.expm1(d) + w_omni * math.expm1(o)
            if sc2 <= sc:
                sc = sc2
            else:
                cur[j] = old
        print(f"  起点 {si + 1}/{len(starts)} 完成，评分 {sc:.6f}")
        if sc < best_score:
            best_score, best_layout = sc, cur.copy()
    return best_layout, best_score


# --------------------------------------------------------------------------
# 4.5 代理引导 + 精英保真的遗传算法（surrogate-assisted GA）
# --------------------------------------------------------------------------
def _edge_miss_fast(layout: np.ndarray, n_azi: int = 90,
                     n_off: int = 24) -> float:
    """贴边对抗缺听（子集版）：g 贴 1770 边缘、径向 ±8° 内偏角、R 三档。

    返回漏例数（归一化 0~1，按总例数）。与 validate() 的口径一致但抽样更密省。
    """
    P = np.vstack([np.zeros(2), np.asarray(layout, dtype=float)])
    miss = 0
    total = 0
    for R in (1000.0, 1250.0, 1500.0):
        for ka in range(n_azi):
            a = 2.0 * math.pi * ka / n_azi
            g = np.array([1770.0 * math.cos(a), 1770.0 * math.sin(a)])
            for ko in range(n_off):
                t = a + math.radians((ko / (n_off - 1) - 0.5) * 8.0)
                d = P - g
                ok = ((np.einsum("ij,ij->i", d, d) <= R * R + 1e-9)
                      & (d[:, 0] * math.cos(t) + d[:, 1] * math.sin(t) >= -1e-9)).any()
                miss += int(not ok)
                total += 1
    return miss / total


def ga_search(net: nn.Module, n_free: int, n_pop: int = 300,
              n_elite: int = 20, n_gen: int = 40, mc_elite: int = 40_000,
              w_omni: float = 0.5, w_edge: float = 0.4, w_route: float = 0.0,
              seed: int = 9, dev: str = "cpu", log_every: int = 5
              ) -> Tuple[np.ndarray, dict]:
    """代理(MNN)引导 + 精英精确验证的进化布局搜索。

    每代：代理对种群排序 → top 精英做蒙特卡洛精评（真实缺听率锚定）→ 精英保留并变异
    （小 σ）+ 交叉生成子代。代理即使对"极好布局"区失真，精英精评也保证种群不漂移，
    最终收敛到真实最优。返回 (最优布局, 过程记录)。
    """
    from cumcm.t4.sweep import measure_layout
    rng = np.random.default_rng(seed)

    def ring(n, r, rot=0.0):
        a = np.linspace(0, 2 * math.pi, n, endpoint=False) + rot
        return np.stack([r * np.cos(a), r * np.sin(a)], 1)

    manual0 = np.asarray(measure_layout()[1:], dtype=float)
    if len(manual0) >= n_free:                 # 点数不足时均匀抽样子集
        idx = np.linspace(0, len(manual0) - 1, n_free).round().astype(int)
        manual = manual0[idx]
    else:
        manual = manual0
    pop = np.empty((n_pop, n_free, 2), dtype=float)
    for i in range(n_pop):
        r = rng.random()
        if r < 0.34:
            pop[i] = manual + rng.normal(0.0, 80.0, manual.shape)      # 手工邻域
        elif r < 0.67:
            nr = int(rng.integers(9, 15)); ni = n_free - nr
            rr = np.full(nr, rng.uniform(1450.0, 1950.0))
            aa = 2.0 * math.pi * np.arange(nr) / nr + rng.uniform(-0.1, 0.1)
            if ni > 0:
                rr = np.concatenate([rr, rng.uniform(250.0, 950.0, ni)])
                aa = np.concatenate([aa, rng.uniform(0.0, 2.0 * math.pi, ni)])
            pop[i, :, 0] = rr * np.cos(aa); pop[i, :, 1] = rr * np.sin(aa)
        else:
            rr = MAX_R * np.sqrt(rng.random(n_free))
            aa = rng.uniform(0.0, 2.0 * math.pi, n_free)
            pop[i, :, 0] = rr * np.cos(aa); pop[i, :, 1] = rr * np.sin(aa)

    feats = np.empty((n_pop, n_free * 2), dtype=np.float32)
    best_overall: Optional[np.ndarray] = None
    best_score = float("inf")
    history = []
    rec: dict = {"history": history, "elite": []}
    for gen in range(n_gen):
        for i in range(n_pop):
            feats[i] = featurize(pop[i])
        with torch.no_grad():
            yc = net(torch.from_numpy(feats).to(dev)).cpu().numpy()
        sc = yc[:, 0] + w_omni * yc[:, 1]
        cand_idx = np.argsort(sc)[:max(n_elite * 3, 40)]     # 代理只初筛候选池
        # 真实精评（定向 + w·全向 + w·贴边对抗）从候选池选精英——真实排序主导
        real = np.array([fast_miss(pop[i], n=mc_elite, seed=seed + gen * 31 + k)
                         for k, i in enumerate(cand_idx)])
        edge = np.array([_edge_miss_fast(pop[i]) for i in cand_idx])
        rt = np.array([route_of(pop[i]) for i in cand_idx]) / 1000.0
        rsc = real[:, 0] + w_omni * real[:, 1] + w_edge * edge + w_route * rt
        order = np.argsort(rsc)[:n_elite]
        elite = cand_idx[order]
        elite_meta = [(float(real[order[k], 0]), float(real[order[k], 1]),
                        float(edge[order[k]]), float(rt[order[k]] * 1000.0))
                      for k in range(n_elite)]
        cur = rsc[order[0]]
        if cur < best_score:
            best_score, best_overall = cur, pop[elite[0]].copy()
        if gen % log_every == 0 or gen == n_gen - 1:
            print(f"  第 {gen + 1}/{n_gen} 代：精英最优 rsc {best_score:.5f}"
                  f"（定向 {real[order[0],0]:.5f} 贴边 {edge[order[0]]:.4f}，"
                  f"候选池 {len(cand_idx)}）")
        history.append(float(best_score))
        rec["elite"] = [(pop[elite[k]].copy(),) + elite_meta[k]
                        for k in range(n_elite)]
        # 生成下一代：精英 20 直传 + 精英变异 + 交叉
        newpop = np.empty_like(pop)
        newpop[:n_elite] = pop[elite]
        k = n_elite
        while k < n_pop:
            if rng.random() < 0.7 and n_elite > 1:
                ia_ = int(elite[rng.integers(0, n_elite)])
                ib_ = int(elite[rng.integers(0, n_elite)])
                mask = rng.random(n_free) < 0.5
                child = np.where(mask[:, None], pop[ia_], pop[ib_])
                child = np.asarray(child, dtype=float).reshape(n_free, 2)
                for j in range(n_free):      # 变异：赋新值而非就地 +=（防视图意外）
                    if rng.random() < 0.25:
                        child[j] = child[j] + rng.normal(0.0, float(rng.uniform(8.0, 45.0)), 2)
            else:
                ip_ = int(elite[rng.integers(0, n_elite)])
                child = (pop[ip_].astype(float)
                         + rng.normal(0.0, float(rng.uniform(8.0, 55.0)), (n_free, 2)))
            rr = np.hypot(child[:, 0], child[:, 1])
            m = rr > MAX_R
            if m.any():
                child[m] *= (MAX_R / np.maximum(rr[m], 1e-9))[:, None]
            newpop[k] = child
            k += 1
        pop = newpop
    rec["best_score"] = best_score
    return best_overall, rec


# --------------------------------------------------------------------------
# 5. 精确验证：高精度 MC + 贴边对抗 + TSP 路程
# --------------------------------------------------------------------------
def route_of(layout: np.ndarray) -> float:
    """从原点出发、访问全部自由点的最短开放路径里程（确定性最近邻+2-opt）。"""
    pts = np.vstack([np.zeros(2), layout])
    D = dist_matrix(pts, (0.0, 0.0))
    order = two_opt_greedy(two_opt_first(nearest_order(pts, start=(0.0, 0.0)), D), D)
    m = 0.0
    prev = pts[0]
    for i in order:
        m += float(np.hypot(*(pts[i] - prev)))
        prev = pts[i]
    return m


def validate(layout: np.ndarray, mc_n: int = 400_000, seed: int = 2026):
    """精确验证：返回 {定向缺听/全向缺听/贴边漏/TSP 路程}。"""
    dm, om = fast_miss(layout, n=mc_n, seed=seed)
    P = np.vstack([np.zeros(2), layout])
    edge = 0
    for R in (1000.0, 1250.0, 1500.0):
        for ka in range(360):
            a = 2.0 * math.pi * ka / 360
            g = np.array([1770.0 * math.cos(a), 1770.0 * math.sin(a)])
            for ko in range(40):
                t = a + math.radians((ko / 39.0 - 0.5) * 8.0)
                d = P - g
                ok = ((np.einsum("ij,ij->i", d, d) <= R * R + 1e-9)
                      & (d[:, 0] * math.cos(t) + d[:, 1] * math.sin(t) >= -1e-9)).any()
                if not ok:
                    edge += 1
    return {"dir_miss": dm, "omni_miss": om, "edge_miss": edge, "route_m": route_of(layout)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--n-free", type=int, default=22)
    ap.add_argument("--mc-n", type=int, default=50_000)
    ap.add_argument("--train-epochs", type=int, default=250)
    ap.add_argument("--candidates", type=int, default=60_000)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--ga-pop", type=int, default=300)
    ap.add_argument("--ga-gen", type=int, default=40)
    ap.add_argument("--ga-elite", type=int, default=20)
    ap.add_argument("--ga-mc", type=int, default=100_000)
    ap.add_argument("--ga-route-w", type=float, default=0.0003)
    ap.add_argument("--validate-n", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", type=str, default="cpu")
    a = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(a.seed)          # 固定网络初始权重与打乱顺序，保证同 seed 可复现

    if a.gen > 0:
        layouts = gen_layouts(a.gen, a.n_free, seed=a.seed)
        labels = label_layouts(layouts, a.mc_n, seed=a.seed + 1)
        np.savez(DATA_PATH, layouts=layouts, labels=labels)
        print(f"数据生成：{a.gen} 布局 × {a.mc_n} 算例，"
              f"定向缺听均值 {labels[:,0].mean():.4f}，全向缺听均值 {labels[:,1].mean():.4f}"
              f"（{time.time()-t0:.0f}s）")
    else:
        try:
            z = np.load(DATA_PATH)
            layouts, labels = z["layouts"], z["labels"]
            print(f"载入缓存数据：{len(layouts)} 布局")
        except FileNotFoundError:
            raise SystemExit("无缓存数据：先 --gen N 生成")

    # 训练目标用 log1p(缺听率)：缺听率跨多个数量级（好布局 1e-4、差布局 0.5），
    # 直接回归则"缺听 0.01% vs 0%（饱和）"无区分度，对数尺度让好布局区也有梯度。
    Y = np.log1p(np.clip(labels, 0.0, 1.0))
    X = np.stack([featurize(l) for l in layouts]).astype(np.float32)
    net, losses = train_net(X, Y, epochs=a.train_epochs, dev=a.device)
    print(f"训练完成（{a.train_epochs} epoch，末损失 {losses[-1]:.5f}，"
          f"{time.time()-t0:.0f}s）")

    with torch.no_grad():
        yh = net(torch.from_numpy(X).to(a.device)).cpu().numpy()
    for k, name in ((0, "定向"), (1, "全向")):
        r = np.corrcoef(Y[:, k], yh[:, k])[0, 1]
        print(f"代理 {name}缺听率相关（log1p 尺度）{r:.4f}")

    # ---- 代理引导 + 精英保真的遗传搜索（替代连续爬山/单轮初筛，鲁棒且收敛）----
    best, rec = ga_search(net, a.n_free, n_pop=a.ga_pop, n_elite=a.ga_elite,
                          n_gen=a.ga_gen, mc_elite=a.ga_mc, w_route=a.ga_route_w,
                          seed=a.seed + 3, dev=a.device)
    best = best.astype(float)
    np.save(BEST_PATH, best)
    if rec.get("elite"):
        el = rec["elite"]
        el_pts = np.stack([e[0] for e in el])
        el_dir = np.array([e[1] for e in el])
        el_route = np.array([e[4] for e in el])
        np.savez("results/t4/.nn_elite.npz", pts=el_pts, dir=el_dir,
                 route=el_route, omni=np.array([e[2] for e in el]),
                 edge=np.array([e[3] for e in el]))
        print("精英榜（前 8，供演练实测虚拟耗时）：")
        for k in range(min(8, len(el))):
            print(f"  #{k + 1}: 定向缺听 {el_dir[k]*100:.4f}% 全向 {el[k][2]*100:.4f}% "
                  f"贴边 {el[k][3]:.0f} 路线 {el_route[k]:.0f} m")

    # ---- 最优布局的最终精确复核（40 万 MC + 贴边对抗 + TSP）----
    v = validate(best, mc_n=a.validate_n)
    print(f"GA 最优（自由 {a.n_free} 点 + 原点 = {a.n_free + 1} 个测量位置）：")
    print(f"  定向听到 {100 - v['dir_miss']*100:.4f}%  全向听到 {100 - v['omni_miss']*100:.4f}%")
    print(f"  贴边对抗漏 {v['edge_miss']}（43 200 例）  TSP 路程 {v['route_m']:.0f} m"
          f"（{time.time()-t0:.0f}s）")
    print("布局（x,y）：")
    for pp in best:
        print(f"  ({pp[0]:.1f}, {pp[1]:.1f})")


if __name__ == "__main__":
    main()