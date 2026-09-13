"""补测选点的 Fisher 信息与交会几何，直接转发 t3.probing"""

from __future__ import annotations

from t3.probing import (ambiguity_area, fisher_sigma, hypothesis_points,
                              probe_candidates)

__all__ = ["fisher_sigma", "hypothesis_points", "probe_candidates", "ambiguity_area"]
