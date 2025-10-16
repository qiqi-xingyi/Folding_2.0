from __future__ import annotations
from typing import Callable, Dict, Any, List, Tuple
import math
import random

def spsa_minimize(fun: Callable[[List[float]], float],
                  x0: List[float],
                  maxiter: int = 200,
                  a: float = 0.1,
                  c: float = 0.1,
                  alpha: float = 0.602,
                  gamma: float = 0.101,
                  seed: int = 42) -> Dict[str, Any]:
    """
    Simple SPSA optimizer for noisy objectives.
    - fun: function mapping parameters -> energy
    - x0: initial parameters
    - schedules: a_k = a / (k+1+1)^alpha, c_k = c / (k+1)^gamma (standard choices)

    Returns dict with 'x', 'fun', 'nit', 'history'.
    """
    rnd = random.Random(seed)
    x = list(x0)
    hist: List[Tuple[int, float]] = []
    best = (x[:], fun(x))

    for k in range(maxiter):
        ak = a / ((k+2) ** alpha)
        ck = c / ((k+1) ** gamma)
        # Rademacher perturbation
        delta = [1 if rnd.random() < 0.5 else -1 for _ in x]
        xp = [xi + ck*di for xi, di in zip(x, delta)]
        xm = [xi - ck*di for xi, di in zip(x, delta)]
        yp = fun(xp)
        ym = fun(xm)
        ghat = [(yp - ym)/(2*ck*di) for di in delta]
        # update
        x = [xi - ak*gi for xi, gi in zip(x, ghat)]
        fx = fun(x)
        hist.append((k, fx))
        if fx < best[1]:
            best = (x[:], fx)

    return {"x": best[0], "fun": best[1], "nit": maxiter, "history": hist}
