"""
optimizer.nsga2
================
A compact, dependency-free (numpy only) NSGA-II implementation:
    - Deb's constrained-dominance (feasible beats infeasible; among
      infeasible, smaller total violation wins; among feasible, normal
      Pareto dominance)
    - Fast non-dominated sort
    - Crowding distance
    - Binary tournament selection (with replacement)
    - Arithmetic (alpha-blend) crossover + Gaussian mutation, with
      bound repair (clip)

It is deliberately generic: it knows nothing about turbojets. You give
it:
    - `bounds`: dict[str, (low, high)]  the design variables to search
    - `evaluate_fn(design: dict) -> (objectives: tuple, constraints: tuple, extra: dict)`
      objectives are MINIMIZED; constraints are satisfied when <= 0.

See optimizer/engine_optimizer.py for the turbojet-specific wiring
(physics model + ML correction + requirement-based objectives).

CHANGES (synced from the updated reference NSGA-II script)
------------------------------------------------------------
1. Crossover is now a pure per-gene arithmetic blend
   (child = alpha*a + (1-alpha)*b), matching the reference script's
   `crossover()`. The previous version additionally did a per-gene
   50/50 "straight swap vs. blend" coin flip; that branch is removed.
2. Tournament selection now samples candidates *with* replacement
   (`random.choice`, like the reference script's `tournament_selection`)
   instead of `random.sample` (without replacement).
3. Feasibility/constraint-dominance now treats any total constraint
   violation <= FEASIBILITY_EPS (1e-12) as feasible, instead of
   requiring an exact `violation == 0.0`. This matches the reference
   script's floating-point-safe feasibility check and avoids a design
   that satisfies every constraint being spuriously marked infeasible
   by a tiny float residue.
No public parameter names changed: this project already uses its own
established nomenclature (`compressor_efficiency`, `turbine_inlet_temp_K`,
`turbine_efficiency`, `mass_flow_kg_s`, ...) everywhere, including in
optimizer/engine_optimizer.py's DESIGN_VARIABLES, so nothing else needed
to be renamed to stay consistent project-wide.
"""

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

# Any total constraint violation at or below this is treated as
# feasible. Guards against a design that truly satisfies every
# constraint (g <= 0 for all g) being marked infeasible by float noise.
FEASIBILITY_EPS = 1e-12


@dataclass
class Individual:
    design: Dict[str, float]
    objectives: Tuple[float, ...] = field(default_factory=tuple)
    constraints: Tuple[float, ...] = field(default_factory=tuple)
    extra: Dict = field(default_factory=dict)
    rank: int = 0
    crowding: float = 0.0

    @property
    def violation(self) -> float:
        return sum(v for v in self.constraints if v > 0)

    @property
    def feasible(self) -> bool:
        return self.violation <= FEASIBILITY_EPS


def constraint_dominates(a: Individual, b: Individual) -> bool:
    va, vb = a.violation, b.violation
    fa, fb = va <= FEASIBILITY_EPS, vb <= FEASIBILITY_EPS
    if fa and not fb:
        return True
    if not fa and fb:
        return False
    if not fa and not fb:
        return va < vb
    return dominates(a, b)


def dominates(a: Individual, b: Individual) -> bool:
    no_worse = all(x <= y for x, y in zip(a.objectives, b.objectives))
    strictly_better = any(x < y for x, y in zip(a.objectives, b.objectives))
    return no_worse and strictly_better


def fast_non_dominated_sort(pop: List[Individual]) -> List[List[Individual]]:
    fronts: List[List[Individual]] = [[]]
    dominated_by = {id(p): [] for p in pop}
    domination_count = {id(p): 0 for p in pop}

    for p in pop:
        for q in pop:
            if p is q:
                continue
            if constraint_dominates(p, q):
                dominated_by[id(p)].append(q)
            elif constraint_dominates(q, p):
                domination_count[id(p)] += 1
        if domination_count[id(p)] == 0:
            p.rank = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_by[id(p)]:
                domination_count[id(q)] -= 1
                if domination_count[id(q)] == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return [f for f in fronts if f]


def crowding_distance(front: List[Individual]) -> None:
    n = len(front)
    if n == 0:
        return
    for p in front:
        p.crowding = 0.0
    if n <= 2:
        # Every member of a front this small is a boundary point on
        # every objective, so give it infinite crowding directly
        # (matches the reference script's explicit short-circuit).
        for p in front:
            p.crowding = float("inf")
        return
    n_obj = len(front[0].objectives)
    for m in range(n_obj):
        front.sort(key=lambda p: p.objectives[m])
        front[0].crowding = float("inf")
        front[-1].crowding = float("inf")
        obj_min = front[0].objectives[m]
        obj_max = front[-1].objectives[m]
        spread = obj_max - obj_min
        if spread == 0:
            continue
        for i in range(1, n - 1):
            front[i].crowding += (
                front[i + 1].objectives[m] - front[i - 1].objectives[m]
            ) / spread


def assign_rank_and_crowding(pop: List[Individual]) -> List[List[Individual]]:
    fronts = fast_non_dominated_sort(pop)
    for front in fronts:
        crowding_distance(front)
    return fronts


def crowded_comparison_better(a: Individual, b: Individual) -> bool:
    if a.rank != b.rank:
        return a.rank < b.rank
    return a.crowding > b.crowding


def tournament_selection(pop: List[Individual], k: int = 2) -> Individual:
    # Sampled WITH replacement (random.choice per pick), matching the
    # reference script's tournament_selection. For the default k=2
    # binary tournament this occasionally compares an individual
    # against itself, in which case it is trivially returned.
    candidates = [random.choice(pop) for _ in range(k)]
    best = candidates[0]
    for c in candidates[1:]:
        if crowded_comparison_better(c, best):
            best = c
    return best


class NSGA2:
    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        evaluate_fn: Callable[[Dict[str, float]], Tuple[Tuple, Tuple, Dict]],
        population_size: int = 80,
        generations: int = 100,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.2,
        mutation_sigma: float = 0.10,
        random_seed: int = 42,
        int_fields: Tuple[str, ...] = (),
        seed_designs: List[Dict[str, float]] = None,
    ):
        self.bounds = bounds
        self.variables = list(bounds.keys())
        self.evaluate_fn = evaluate_fn
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_sigma = mutation_sigma
        self.int_fields = set(int_fields)
        # Optional known-good design points (e.g. a reference baseline)
        # injected into the initial population instead of a random
        # individual, so the search starts with at least one evaluated,
        # physically-sensible point. Never affects behaviour when left
        # at the default (None / empty).
        self.seed_designs = list(seed_designs or [])

        random.seed(random_seed)
        np.random.seed(random_seed)

    # -- design vector helpers -------------------------------------
    def _random_design(self) -> Dict[str, float]:
        d = {}
        for v in self.variables:
            lo, hi = self.bounds[v]
            val = random.uniform(lo, hi)
            d[v] = round(val) if v in self.int_fields else val
        return d

    def _clip(self, design: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for v in self.variables:
            lo, hi = self.bounds[v]
            val = max(lo, min(hi, design[v]))
            out[v] = round(val) if v in self.int_fields else val
        return out

    def _evaluate(self, design: Dict[str, float]) -> Individual:
        design = self._clip(design)
        try:
            objectives, constraints, extra = self.evaluate_fn(design)
        except Exception:  # noqa: BLE001 - invalid design point, penalize heavily
            objectives = tuple([1e12] * 5)
            constraints = tuple([1e12])
            extra = {"invalid": True}
        return Individual(design=design, objectives=objectives, constraints=constraints, extra=extra)

    # -- genetic operators --------------------------------------------
    def _crossover(self, a: Dict[str, float], b: Dict[str, float]):
        # Pure per-gene arithmetic (alpha-blend) crossover: a fresh
        # random alpha per variable, child1 = alpha*a + (1-alpha)*b and
        # child2 the complementary blend. Gated once per parent pair by
        # crossover_prob; below that probability the children are
        # exact copies of the parents (still subject to mutation).
        if random.random() < self.crossover_prob:
            child1, child2 = {}, {}
            for v in self.variables:
                alpha = random.random()
                child1[v] = alpha * a[v] + (1 - alpha) * b[v]
                child2[v] = alpha * b[v] + (1 - alpha) * a[v]
        else:
            child1, child2 = dict(a), dict(b)
        return self._clip(child1), self._clip(child2)

    def _mutate(self, design: Dict[str, float]) -> Dict[str, float]:
        out = dict(design)
        for v in self.variables:
            if random.random() < self.mutation_prob:
                lo, hi = self.bounds[v]
                span = hi - lo
                out[v] = out[v] + random.gauss(0, self.mutation_sigma * span)
        return self._clip(out)

    # -- main loop -------------------------------------------------
    def _initial_population(self) -> List[Individual]:
        n_seed = min(len(self.seed_designs), self.population_size)
        pop = [self._evaluate(dict(d)) for d in self.seed_designs[:n_seed]]
        pop += [self._evaluate(self._random_design()) for _ in range(self.population_size - n_seed)]
        return pop

    def _create_offspring(self, pop: List[Individual]) -> List[Individual]:
        offspring = []
        while len(offspring) < self.population_size:
            p1 = tournament_selection(pop)
            p2 = tournament_selection(pop)
            c1, c2 = self._crossover(p1.design, p2.design)
            c1, c2 = self._mutate(c1), self._mutate(c2)
            offspring.append(self._evaluate(c1))
            if len(offspring) < self.population_size:
                offspring.append(self._evaluate(c2))
        return offspring

    def _environmental_selection(self, combined: List[Individual]) -> List[Individual]:
        fronts = assign_rank_and_crowding(combined)
        new_pop: List[Individual] = []
        for front in fronts:
            if len(new_pop) + len(front) <= self.population_size:
                new_pop.extend(front)
            else:
                remaining = self.population_size - len(new_pop)
                front_sorted = sorted(front, key=lambda p: -p.crowding)
                new_pop.extend(front_sorted[:remaining])
                break
        return new_pop

    def run(self, verbose: bool = True) -> List[Individual]:
        pop = self._initial_population()
        assign_rank_and_crowding(pop)

        for gen in range(self.generations):
            offspring = self._create_offspring(pop)
            combined = pop + offspring
            pop = self._environmental_selection(combined)

            if verbose and (gen % max(1, self.generations // 10) == 0 or gen == self.generations - 1):
                n_feasible = sum(1 for p in pop if p.feasible)
                print(f"  gen {gen+1:4d}/{self.generations}  feasible: {n_feasible}/{len(pop)}")

        return pop

    @staticmethod
    def pareto_front(pop: List[Individual]) -> List[Individual]:
        return [p for p in pop if p.rank == 0]
