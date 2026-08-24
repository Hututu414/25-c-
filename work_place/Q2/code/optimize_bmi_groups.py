from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.stats import beta as beta_distribution

THRESHOLD = 0.04


def ga_to_week_day(ga: float) -> str:
    if not np.isfinite(ga):
        return "未达到"
    week = int(np.floor(ga))
    day = int(np.rint((ga - week) * 7))
    if day == 7:
        week, day = week + 1, 0
    return f"{week}周+{day}天"


def probability_kernel(
    eta_grid: np.ndarray,
    phi: float,
    measurement_sd: float,
    error_nodes: int = 21,
) -> np.ndarray:
    mu = np.clip(1.0 / (1.0 + np.exp(-eta_grid)), 1e-12, 1 - 1e-12)
    alpha, beta = mu * phi, (1.0 - mu) * phi
    if measurement_sd == 0:
        return beta_distribution.sf(THRESHOLD, alpha, beta)
    nodes, weights = hermgauss(error_nodes)
    errors = np.sqrt(2.0) * measurement_sd * nodes
    normalized = weights / np.sqrt(np.pi)
    result = np.zeros_like(eta_grid, dtype=float)
    for error, weight in zip(errors, normalized, strict=True):
        cutoff = THRESHOLD - error
        if cutoff <= 0:
            result += weight
        elif cutoff < 1:
            result += weight * beta_distribution.sf(cutoff, alpha, beta)
    return result


@dataclass(frozen=True)
class ProbabilityEngine:
    eta_grid: np.ndarray
    kernels: dict[str, np.ndarray]
    u_nodes: np.ndarray
    u_weights: np.ndarray
    empirical_offsets: np.ndarray

    @classmethod
    def create(
        cls,
        phi: float,
        sigma_u2: float,
        empirical_offsets: np.ndarray,
        eta_min: float,
        eta_max: float,
        measurement_sds: dict[str, float],
        eta_step: float = 0.0005,
    ) -> ProbabilityEngine:
        eta_grid = np.arange(eta_min - 0.01, eta_max + 0.01 + eta_step, eta_step)
        kernels = {
            name: probability_kernel(eta_grid, phi, sd)
            for name, sd in measurement_sds.items()
        }
        nodes, weights = hermgauss(30)
        u_nodes = np.sqrt(2.0 * sigma_u2) * nodes
        u_weights = weights / np.sqrt(np.pi)
        return cls(
            eta_grid=eta_grid,
            kernels=kernels,
            u_nodes=u_nodes,
            u_weights=u_weights,
            empirical_offsets=np.asarray(empirical_offsets, dtype=float),
        )

    def _mixture(self, covariates: dict[str, np.ndarray] | None) -> tuple[np.ndarray, np.ndarray]:
        if covariates is None:
            rounded = np.round(self.empirical_offsets, 12)
            offsets, counts = np.unique(rounded, return_counts=True)
            offset_weights = counts / counts.sum()
        else:
            if "z_offsets" not in covariates:
                raise ValueError("covariates must provide R-derived z_offsets")
            offsets = np.atleast_1d(np.asarray(covariates["z_offsets"], dtype=float))
            offset_weights = np.atleast_1d(
                np.asarray(covariates.get("weights", np.ones(len(offsets))), dtype=float)
            )
            if len(offsets) != len(offset_weights) or np.any(offset_weights < 0):
                raise ValueError("invalid covariate offset weights")
            offset_weights = offset_weights / offset_weights.sum()
        shifts = (offsets[:, None] + self.u_nodes[None, :]).ravel()
        weights = (offset_weights[:, None] * self.u_weights[None, :]).ravel()
        return shifts, weights

    def evaluate_eta(
        self,
        eta_base: np.ndarray,
        scenario: str = "no_error",
        covariates: dict[str, np.ndarray] | None = None,
        chunk_size: int = 256,
    ) -> np.ndarray:
        if scenario not in self.kernels:
            raise KeyError(f"unknown reliability scenario: {scenario}")
        eta_base = np.asarray(eta_base, dtype=float)
        shifts, weights = self._mixture(covariates)
        output = np.empty_like(eta_base)
        kernel = self.kernels[scenario]
        for start in range(0, len(eta_base), chunk_size):
            stop = min(start + chunk_size, len(eta_base))
            eta = eta_base[start:stop, None] + shifts[None, :]
            probabilities = np.interp(eta.ravel(), self.eta_grid, kernel).reshape(eta.shape)
            output[start:stop] = probabilities @ weights
        return output

    def reliability_curve(
        self,
        bmi: float,
        eta_surface: pd.DataFrame,
        covariates: dict[str, np.ndarray] | None = None,
        scenario: str = "no_error",
    ) -> pd.DataFrame:
        """Q3-ready interface: future covariates enter only through R-derived z_offsets."""
        pivot = eta_surface.pivot(index="GA", columns="BMI", values="eta_base").sort_index().sort_index(axis=1)
        eta = np.array(
            [np.interp(bmi, pivot.columns.to_numpy(float), row) for row in pivot.to_numpy(float)],
            dtype=float,
        )
        return pd.DataFrame(
            {
                "GA": pivot.index.to_numpy(float),
                "BMI": bmi,
                "scenario": scenario,
                "reliability": self.evaluate_eta(eta, scenario, covariates),
            }
        )


@dataclass(frozen=True)
class Segment:
    start: int
    stop: int
    n_patients: int
    bmi_min: float
    bmi_max: float
    cost: float


@dataclass(frozen=True)
class SegmentationResult:
    k: int
    objective: float
    segments: tuple[Segment, ...]

    @property
    def cutpoints(self) -> tuple[float, ...]:
        return tuple(
            (self.segments[index].bmi_max + self.segments[index + 1].bmi_min) / 2
            for index in range(len(self.segments) - 1)
        )


def optimal_segmentations(
    patient_bmi: np.ndarray,
    patient_curves: np.ndarray,
    ga_grid: np.ndarray,
    k_values: tuple[int, ...] = (2, 3, 4, 5),
    min_group_size: int = 30,
) -> dict[int, SegmentationResult]:
    patient_bmi = np.asarray(patient_bmi, dtype=float)
    patient_curves = np.asarray(patient_curves, dtype=float)
    ga_grid = np.asarray(ga_grid, dtype=float)
    if patient_curves.shape != (len(patient_bmi), len(ga_grid)):
        raise ValueError("patient curve matrix has an invalid shape")
    order = np.argsort(patient_bmi, kind="stable")
    bmi_sorted, curves_sorted = patient_bmi[order], patient_curves[order]
    unique_bmi, first, counts = np.unique(bmi_sorted, return_index=True, return_counts=True)
    block_curves = np.vstack(
        [curves_sorted[start : start + count].mean(axis=0) for start, count in zip(first, counts, strict=True)]
    )
    trapezoid = np.full(len(ga_grid), float(np.diff(ga_grid).mean()))
    trapezoid[[0, -1]] *= 0.5
    transformed = block_curves * np.sqrt(trapezoid)
    prefix_curve = np.vstack([np.zeros(len(ga_grid)), np.cumsum(counts[:, None] * transformed, axis=0)])
    prefix_norm = np.r_[0.0, np.cumsum(counts * np.einsum("ij,ij->i", transformed, transformed))]
    prefix_n = np.r_[0, np.cumsum(counts)]
    block_count = len(unique_bmi)

    def interval_cost(starts: np.ndarray, stop: int) -> np.ndarray:
        totals = prefix_curve[stop] - prefix_curve[starts]
        sizes = prefix_n[stop] - prefix_n[starts]
        norms = prefix_norm[stop] - prefix_norm[starts]
        return np.maximum(norms - np.einsum("ij,ij->i", totals, totals) / sizes, 0.0)

    results: dict[int, SegmentationResult] = {}
    max_k = max(k_values)
    dp = np.full((max_k + 1, block_count + 1), np.inf)
    previous = np.full((max_k + 1, block_count + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for k in range(1, max_k + 1):
        for stop in range(1, block_count + 1):
            starts = np.arange(stop)
            sizes = prefix_n[stop] - prefix_n[starts]
            valid = (sizes >= min_group_size) & np.isfinite(dp[k - 1, starts])
            if not np.any(valid):
                continue
            candidate_starts = starts[valid]
            values = dp[k - 1, candidate_starts] + interval_cost(candidate_starts, stop)
            best = int(np.argmin(values))
            dp[k, stop] = values[best]
            previous[k, stop] = candidate_starts[best]
        if k not in k_values or not np.isfinite(dp[k, block_count]):
            continue
        boundaries: list[tuple[int, int]] = []
        stop = block_count
        for level in range(k, 0, -1):
            start = int(previous[level, stop])
            if start < 0:
                raise RuntimeError("segmentation backtracking failed")
            boundaries.append((start, stop))
            stop = start
        boundaries.reverse()
        segments = tuple(
            Segment(
                start=start,
                stop=stop,
                n_patients=int(prefix_n[stop] - prefix_n[start]),
                bmi_min=float(unique_bmi[start]),
                bmi_max=float(unique_bmi[stop - 1]),
                cost=float(interval_cost(np.array([start]), stop)[0]),
            )
            for start, stop in boundaries
        )
        results[k] = SegmentationResult(k=k, objective=float(dp[k, block_count]), segments=segments)
    if set(results) != set(k_values):
        raise RuntimeError(f"no feasible DP solution for K={sorted(set(k_values) - set(results))}")
    return results


def choose_k(results: dict[int, SegmentationResult]) -> int:
    ks = np.array(sorted(results), dtype=int)
    objectives = np.array([results[int(k)].objective for k in ks], dtype=float)
    gain_to_three = (objectives[0] - objectives[1]) / objectives[0]
    if gain_to_three < 0.10:
        return int(ks[0])
    x = (ks - ks[0]) / (ks[-1] - ks[0])
    y = (np.log(objectives) - np.log(objectives[-1])) / (
        np.log(objectives[0]) - np.log(objectives[-1])
    )
    distance = np.abs(y - (1 - x)) / np.sqrt(2)
    return int(ks[np.argmax(distance)])


def assign_groups(bmi: np.ndarray, result: SegmentationResult) -> np.ndarray:
    return np.searchsorted(np.asarray(result.cutpoints), np.asarray(bmi, dtype=float), side="right") + 1


def timing_audit(
    profiles: pd.DataFrame,
    curves: np.ndarray,
    ga_grid: np.ndarray,
    result: SegmentationResult,
    q: float,
    scenario: str,
) -> pd.DataFrame:
    groups = assign_groups(profiles["BMI"].to_numpy(float), result)
    rows = []
    for group_id, segment in enumerate(result.segments, 1):
        selected = groups == group_id
        group_curves = curves[selected]
        mean_curve = group_curves.mean(axis=0)
        eligible = np.flatnonzero(mean_curve >= q)
        timing_index = int(eligible[0]) if len(eligible) else -1
        if timing_index >= 0:
            values = group_curves[:, timing_index]
            timing = float(ga_grid[timing_index])
            mean_reliability = float(values.mean())
            q10 = float(np.quantile(values, 0.10))
            minimum = float(values.min())
            proportion = float(np.mean(values >= q))
        else:
            timing = mean_reliability = q10 = minimum = proportion = np.nan
        lower = segment.bmi_min if group_id == 1 else result.cutpoints[group_id - 2]
        upper = segment.bmi_max if group_id == result.k else result.cutpoints[group_id - 1]
        interval = f"[{lower:.2f}, {upper:.2f}]" if group_id == 1 else f"({lower:.2f}, {upper:.2f}]"
        rows.append(
            {
                "scenario": scenario,
                "q": q,
                "Group": group_id,
                "BMI_interval": interval,
                "BMI_lower": lower,
                "BMI_upper": upper,
                "N": int(selected.sum()),
                "group_curve_cost": segment.cost,
                "recommended_GA": timing,
                "week_day": ga_to_week_day(timing),
                "mean_reliability": mean_reliability,
                "q10_reliability": q10,
                "min_reliability": minimum,
                "proportion_at_or_above_q": proportion,
                "any_below_q": bool(np.any(values < q)) if timing_index >= 0 else True,
                "marked_tail_failure": bool(minimum < q - 0.05) if timing_index >= 0 else True,
            }
        )
    return pd.DataFrame(rows)


def self_check() -> None:
    assert ga_to_week_day(12.0) == "12周+0天"
    assert ga_to_week_day(12.9) == "12周+6天"
    bmi = np.repeat([20.0, 25.0, 30.0, 35.0], 30)
    ga = np.array([10.0, 10.1, 10.2])
    curves = np.repeat(np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.7, 0.8, 0.9], [0.7, 0.8, 0.9]]), 30, axis=0)
    result = optimal_segmentations(bmi, curves, ga, k_values=(2,), min_group_size=30)[2]
    assert result.k == 2 and [segment.n_patients for segment in result.segments] == [60, 60]


if __name__ == "__main__":
    self_check()
    print("self-check passed")
