"""nn_layout_speed.py：用神经网络找"耗时更短"的测量点布局（保持检测保证）。

前一轮（nn_layout.py 自由搜索）已证明：**听率是隐性时间成本**——自由布局听率
99.8% 时，漏掉的源并非均匀小概率而是特定区域系统性覆盖缺失，导致收尾专程移动暴涨
（实测总里程 28~32 km > 人工 26.6 km），即使路线短 300 m 也净亏。

本模块修正目标函数：**固定 7 个覆盖基点**（T1/T3 覆盖圆，保证全向覆盖与内部定向覆盖），
只让神经网络/遗传算法优化**外圈点摆位**（点数、半径、方位连续可调）。这样：
  * 听率天然 ≥99.9% 级（基点保证），漏源是真正的小概率，收尾不暴涨；
  * 可压缩的真实成本 = 扫描路线（TSP）+ 测量点数（no_signal 试探）+ 顺路清除绕路，
    全部进入精英精评目标 → NN 找到比人工"12 均匀外圈"更省时的摆位。

运行（示例）：
    python -X utf8 -m cumcm.t4.nn_layout_speed --gen 2500 --n-outer 12 --ga-outer 11 \\
        --ga-gen 30 --ga-mc 200000 --validate-n 400000
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
from cumcm.t3.config import SURVEY_CENTERS

MAX_R = 2270.0
OUTER_SLOT = 12            # 代理输入固定 12 个外圈点（不足补 0、多余截断）
DATA_PATH = "results/t4/.nn_speed_data.npz"
BEST_PATH = "results/t4/.nn_speed_best.npy"
ELITE_PATH = "results/t4/.nn_speed_elite.npz"


def base7() -> np.ndarray:
    """7 个覆盖基点（T1/T3 覆盖圆的圆心，固定）。"""
    return np.asarray(SURVEY_CENTERS, dtype=float)


def assemble(outer: np.ndarray) -> np.ndarray:
    """外圈点 → 完整布局（原点 + 7 基点 + 外圈点）。outer 为 (k,2)，k=实际点数。"""
    return np.vstack([np.zeros(2), base7(), np.asarray(outer, dtype=float)])


# --------------------------------------------------------------------------
# 向量化蒙特卡洛缺听率（含 7 基点结构）
# --------------------------------------------------------------------------
def fast_miss(pts: Sequence[Sequence[float]], n: int = 30_000,
              seed: int = 0) -> Tuple[float, float]:
    P = np.asarray(pts, dtype=float)
    rng = np.random.default_rng(seed)
    g_r = 1770.0 * np.sqrt(rng.random(n))
    g_a = rng.random(n) * 2.0 * math.pi
    G = np.stack((g_r * np.cos(g_a), g_r * np.sin(g_a)), axis=1)
    th = rng.random(n) * 2.0 * math.pi
    U = np.stack((np.cos(th), np.sin(th)), axis=1)
    Rv = rng.uniform(1000.0, 1500.0, n)
    D = G[:, None, :] - P[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", D, D)
    in_rad = dist2 <= Rv[:, None] * Rv[:, None] + 1e-9
    proj = np.einsum("ijk,ik->ij", D, U)
    dir_hit = in_rad & (proj >= -1e-9)
    return 1.0 - dir_hit.any(axis=1).mean(), 1.0 - in_rad.any(axis=1).mean()


def _edge_miss_fast(outer: np.ndarray, n_azi: int = 90, n_off: int = 24) -> float:
    P = assemble(outer)
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


# --------------------------------------------------------------------------
# 数据生成：外圈点（≤12）布局 → 含 7 基点的缺听率标签
# --------------------------------------------------------------------------
def gen_outer(n: int, seed: int = 1) -> np.ndarray:
    """生成 n 个外圈配置（12 槽位，实际 8~12 个点，补 0），标签用 assemble+fast_miss。"""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, OUTER_SLOT, 2), dtype=float)
    for i in range(n):
        k = int(rng.integers(8, 13))                     # 本轮实际外圈点数
        r = rng.random()
        if r < 0.6:                                       # 均匀环 + 轻微抖动
            rr = rng.uniform(1600.0, 2100.0)
            aa = 2.0 * math.pi * np.arange(k) / k + rng.uniform(-0.08, 0.08)
            pts = np.stack([rr * np.cos(aa), rr * np.sin(aa)], 1)
            pts += rng.normal(0.0, 40.0, pts.shape)
        else:                                             # 自由摆位（含贴边/偏移）
            rr = np.sqrt(rng.random(k)) * 2270.0
            aa = rng.uniform(0.0, 2.0 * math.pi, k)
            if rng.random() < 0.5:                        # 局部加密一两个扇区
                aa[:2] = rng.uniform(0.0, 2.0 * math.pi) + np.array([0.0, 0.35])
            pts = np.stack([rr * np.cos(aa), rr * np.sin(aa)], 1)
        out[i, :k] = pts
    return out


def label_outer(configs: np.ndarray, mc_n: int, seed: int = 7) -> np.ndarray:
    M = len(configs)
    lab = np.empty((M, 2), dtype=float)
    for i in range(M):
        k = int((np.abs(configs[i]).sum(axis=1) > 1e-9).sum())
        lab[i] = fast_miss(assemble(configs[i, :k]), n=mc_n, seed=seed + i * 7)
    return lab


# --------------------------------------------------------------------------
# 代理（输入 12 槽位外圈点，输出 log1p 缺听）
# --------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, n_in: int, h1: int = 128, h2: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, h1), nn.ReLU(),
                                 nn.Linear(h1, h2), nn.ReLU(),
                                 nn.Linear(h2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def featurize(outer: np.ndarray) -> np.ndarray:
    """(12,2) 外圈槽位 → (24,) 特征（/MAX_R 归一）。"""
    return (outer.reshape(-1) / MAX_R).astype(np.float32)


def train_net(X: np.ndarray, Y: np.ndarray, epochs: int = 200,
              batch: int = 256, lr: float = 1.5e-3, dev: str = "cpu"
              ) -> Tuple[nn.Module, List[float]]:
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
# GA：固定 7 基点，优化 k 个外圈点（k 由 --ga-outer 给定）
# --------------------------------------------------------------------------
def route_of(layout: np.ndarray) -> float:
    pts = np.asarray(layout, dtype=float)
    D = dist_matrix(pts, (0.0, 0.0))
    order = two_opt_greedy(two_opt_first(nearest_order(pts, start=(0.0, 0.0)), D), D)
    m = 0.0
    prev = pts[0]
    for i in order:
        m += float(np.hypot(*(pts[i] - prev)))
        prev = pts[i]
    return m


def ga_outer(net: nn.Module, k_outer: int, n_pop: int = 200, n_elite: int = 12,
             n_gen: int = 30, mc_elite: int = 200_000, w_omni: float = 0.5,
             w_edge: float = 0.4, w_route: float = 0.0003, seed: int = 9,
             dev: str = "cpu", log_every: int = 5) -> Tuple[np.ndarray, dict]:
    """代理引导 + 精英保真的进化搜索：个体 = k 个外圈点。

    精英精评目标 = 定向缺听 + w·全向 + w·贴边 + w·路线/1000（听率主导、路线平局打破）。
    返回 (最优外圈点, 过程记录含精英榜)。
    """
    rng = np.random.default_rng(seed)
    B = base7()
    # 初始种群：均匀环 + 扰动 / 自由点
    pop = np.empty((n_pop, k_outer, 2), dtype=float)
    for i in range(n_pop):
        if rng.random() < 0.7:
            rr = rng.uniform(1650.0, 2050.0)
            aa = 2.0 * math.pi * np.arange(k_outer) / k_outer + rng.uniform(-0.1, 0.1)
            pop[i] = np.stack([rr * np.cos(aa), rr * np.sin(aa)], 1)
            pop[i] += rng.normal(0.0, 50.0, pop[i].shape)
        else:
            rrr = np.sqrt(rng.random(k_outer)) * 2270.0
            aaa = rng.uniform(0.0, 2.0 * math.pi, k_outer)
            pop[i, :, 0] = rrr * np.cos(aaa)
            pop[i, :, 1] = rrr * np.sin(aaa)
    feats = np.empty((n_pop, OUTER_SLOT * 2), dtype=np.float32)
    best: Optional[np.ndarray] = None
    best_score = float("inf")
    rec: dict = {"history": [], "elite": []}
    for gen in range(n_gen):
        for i in range(n_pop):
            f = np.zeros((OUTER_SLOT, 2), dtype=float)
            f[:k_outer] = pop[i]
            feats[i] = featurize(f)
        with torch.no_grad():
            yc = net(torch.from_numpy(feats).to(dev)).cpu().numpy()
        sc = yc[:, 0] + w_omni * yc[:, 1]
        cand_idx = np.argsort(sc)[:max(n_elite * 3, 36)]
        dirs = np.empty(len(cand_idx))
        omnis = np.empty(len(cand_idx))
        edges = np.empty(len(cand_idx))
        rts = np.empty(len(cand_idx))
        rsc = np.empty(len(cand_idx))
        for j, i in enumerate(cand_idx):
            lay = assemble(pop[i])
            dm, om = fast_miss(lay, n=mc_elite, seed=seed + gen * 101 + j)
            eg = _edge_miss_fast(pop[i])
            rt = route_of(lay) / 1000.0
            dirs[j], omnis[j], edges[j], rts[j] = dm, om, eg, rt
            rsc[j] = dm + w_omni * om + w_edge * eg + w_route * rt
        order = np.argsort(rsc)[:n_elite]
        elite = cand_idx[order]
        rec["elite"] = [(pop[elite[k]].copy(), float(dirs[order[k]]),
                         float(omnis[order[k]]), float(edges[order[k]]),
                         float(rts[order[k]] * 1000.0)) for k in range(n_elite)]
        cur = rsc[order[0]]
        if cur < best_score:
            best_score, best = cur, pop[elite[0]].copy()
        if gen % log_every == 0 or gen == n_gen - 1:
            print(f"  第 {gen + 1}/{n_gen} 代：精英最优 rsc {best_score:.5f}"
                  f"（定向 {dirs[order[0]]*100:.4f}% 贴边 {edges[order[0]]:.4f} "
                  f"路线 {rts[order[0]]*1000:.0f}m）")
        rec["history"].append(float(best_score))
        newpop = np.empty_like(pop)
        newpop[:n_elite] = pop[elite]
        k2 = n_elite
        while k2 < n_pop:
            if rng.random() < 0.7:
                ia_ = int(elite[rng.integers(0, n_elite)])
                ib_ = int(elite[rng.integers(0, n_elite)])
                mask = rng.random(k_outer) < 0.5
                child = np.where(mask[:, None], pop[ia_], pop[ib_])
                child = np.asarray(child, dtype=float).reshape(k_outer, 2)
                for j in range(k_outer):
                    if rng.random() < 0.25:
                        child[j] = child[j] + rng.normal(0.0, float(rng.uniform(8.0, 45.0)), 2)
            else:
                ip_ = int(elite[rng.integers(0, n_elite)])
                child = (pop[ip_].astype(float)
                         + rng.normal(0.0, float(rng.uniform(8.0, 55.0)), (k_outer, 2)))
            rr = np.hypot(child[:, 0], child[:, 1])
            msk = rr > MAX_R
            if msk.any():
                child[msk] *= (MAX_R / np.maximum(rr[msk], 1e-9))[:, None]
            newpop[k2] = child
            k2 += 1
        pop = newpop
    rec["best_score"] = best_score
    return best, rec


def validate(outer: np.ndarray, mc_n: int = 400_000, seed: int = 2026):
    lay = assemble(outer)
    dm, om = fast_miss(lay, n=mc_n, seed=seed)
    P = lay
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
                edge += int(not ok)
    return {"dir_miss": dm, "omni_miss": om, "edge_miss": edge, "route_m": route_of(lay)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--mc-n", type=int, default=30_000)
    ap.add_argument("--train-epochs", type=int, default=200)
    ap.add_argument("--ga-outer", type=int, default=11)
    ap.add_argument("--ga-pop", type=int, default=200)
    ap.add_argument("--ga-gen", type=int, default=30)
    ap.add_argument("--ga-elite", type=int, default=12)
    ap.add_argument("--ga-mc", type=int, default=200_000)
    ap.add_argument("--ga-route-w", type=float, default=0.0003)
    ap.add_argument("--validate-n", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", type=str, default="cpu")
    a = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(a.seed)

    if a.gen > 0:
        cfg = gen_outer(a.gen, seed=a.seed)
        lab = label_outer(cfg, a.mc_n, seed=a.seed + 1)
        np.savez(DATA_PATH, cfg=cfg, lab=lab)
        print(f"数据生成：{a.gen} 外圈配置 × {a.mc_n} 算例，"
              f"定向缺听均值 {lab[:,0].mean():.4f}，全向 {lab[:,1].mean():.4f}"
              f"（{time.time()-t0:.0f}s）")
    else:
        z = np.load(DATA_PATH)
        cfg, lab = z["cfg"], z["lab"]
        print(f"载入数据：{len(cfg)} 配置")

    Y = np.log1p(np.clip(lab, 0.0, 1.0))
    X = np.stack([featurize(c) for c in cfg]).astype(np.float32)
    net, losses = train_net(X, Y, epochs=a.train_epochs, dev=a.device)
    print(f"训练完成（{a.train_epochs} epoch，末损失 {losses[-1]:.5f}）")
    with torch.no_grad():
        yh = net(torch.from_numpy(X).to(a.device)).cpu().numpy()
    for k, name in ((0, "定向"), (1, "全向")):
        r = np.corrcoef(Y[:, k], yh[:, k])[0, 1]
        print(f"代理 {name}缺听率相关 "
              f"{r:.4f}" if not np.isnan(r) else f"代理 {name}缺听率相关 nan(标签全 0)")

    best, rec = ga_outer(net, a.ga_outer, n_pop=a.ga_pop, n_elite=a.ga_elite,
                         n_gen=a.ga_gen, mc_elite=a.ga_mc, w_route=a.ga_route_w,
                         seed=a.seed + 3, dev=a.device)
    np.save(BEST_PATH, best)
    el = rec["elite"]
    np.savez(ELITE_PATH, pts=np.stack([e[0] for e in el]),
             dir=np.array([e[1] for e in el]), omni=np.array([e[2] for e in el]),
             edge=np.array([e[3] for e in el]), route=np.array([e[4] for e in el]))
    print(f"GA 完成：{a.ga_outer} 外圈点（+7 基点 + 原点 = {a.ga_outer + 8} 位置）")
    print("精英榜（前 10，供演练实测）：")
    for k in range(min(10, len(el))):
        print(f"  #{k + 1}: 定向缺听 {el[k][1]*100:.4f}% 贴边 {el[k][3]:.0f} "
              f"路线 {el[k][4]:.0f} m")
    v = validate(best, mc_n=a.validate_n)
    print(f"最优验证：定向听到 {100 - v['dir_miss']*100:.4f}% 全向 {100 - v['omni_miss']*100:.4f}% "
          f"贴边 {v['edge_miss']} TSP {v['route_m']:.0f} m（{time.time()-t0:.0f}s）")


if __name__ == "__main__":
    main()