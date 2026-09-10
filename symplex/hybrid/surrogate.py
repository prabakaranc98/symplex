"""Learned components that report where they were trained and how wrong they are there.

Two surrogates, both implemented directly in numpy/scipy:

* `GPEmulator` - Gaussian process regression (Rasmussen and Williams, "Gaussian Processes
  for Machine Learning", MIT Press 2006) with RBF and Matern-5/2 kernels, a learned
  nugget, and marginal-likelihood hyperparameter optimization with analytic gradients.
  A GP is used where uncertainty matters because its posterior variance comes from the
  same fit as its mean, rather than being bolted on afterwards.
* `NeuralSurrogate` - a small multilayer perceptron trained by L-BFGS-B over the flattened
  weights, with backpropagation gradients implemented here and checkable against finite
  differences via `NeuralSurrogate.gradient_check`.

The audit requirement this module exists to satisfy: a neural surrogate coupled to a
mechanistic model must expose its training support and its approximation error at that
coupling. `TrainingSupport` classifies every prediction as interior, boundary or
extrapolation and reports a distance, and every `predict` call carries that verdict with
the value. A GP posterior variance is a statement inside the model's own assumptions; it
is not a bound on error under extrapolation, and `approximation_error` reports held-out
error conditional on support status because the aggregate number hides the difference.
"""

import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, QhullError

from symplex.core.contracts import Invalid

MAX_SURROGATE_SAMPLES = 100_000
MAX_INPUT_DIMENSION = 32
MAX_OUTPUT_DIMENSION = 8
# A GP fit is a Cholesky factorization of an n-by-n matrix: O(n^3) time, O(n^2) memory.
MAX_GP_POINTS = 400
MAX_MLP_PARAMETERS = 20_000
MAX_HIDDEN_LAYERS = 4
MAX_HIDDEN_UNITS = 256
MAX_OPTIMIZER_ITERATIONS = 5_000
MAX_SUPPORT_POINTS = 20_000
MAX_HULL_DIMENSION = 6

KERNELS = ("rbf", "matern52")
ACTIVATIONS = ("tanh", "relu")
SUPPORT_STATUSES = ("interior", "boundary", "extrapolation")

SCOPE = (
    "A fitted approximation of the supplied samples. Accuracy is established only where "
    "training inputs and held-out inputs actually lie; nothing here bounds behaviour "
    "outside the reported training support."
)


def _matrix(values, label, max_cols=MAX_INPUT_DIMENSION, max_rows=MAX_SURROGATE_SAMPLES):
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.size == 0:
        raise Invalid(f"{label} must be a nonempty 1-D or 2-D numeric array")
    if not np.all(np.isfinite(array)):
        raise Invalid(f"{label} contains nonfinite values")
    if array.shape[0] > max_rows:
        raise Invalid(f"{label} exceeds the {max_rows}-row limit")
    if array.shape[1] > max_cols:
        raise Invalid(f"{label} exceeds the {max_cols}-column limit")
    return array


def _integer(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise Invalid(f"{label} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise Invalid(f"{label} must lie in [{low}, {high}]")
    return value


def _number(value, label, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise Invalid(f"{label} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise Invalid(f"{label} must be finite")
    if low is not None and value < low:
        raise Invalid(f"{label} must be at least {low}")
    if high is not None and value > high:
        raise Invalid(f"{label} must be at most {high}")
    return value


def _scaled_square_differences(A, B, length_scale):
    difference = (A[:, None, :] - B[None, :, :]) / length_scale[None, None, :]
    return difference * difference


def _kernel_from_squares(squares, variance, kernel):
    total = squares.sum(axis=-1)
    if kernel == "rbf":
        K = variance * np.exp(-0.5 * total)
        return K, K  # dK/dlog(length_scale_d) = factor * squares[..., d]
    root5r = math.sqrt(5.0) * np.sqrt(np.maximum(total, 0.0))
    decay = np.exp(-root5r)
    K = variance * (1.0 + root5r + root5r * root5r / 3.0) * decay
    factor = (5.0 / 3.0) * variance * (1.0 + root5r) * decay
    return K, factor


def kernel_matrix(A, B, kernel="matern52", length_scale=1.0, variance=1.0):
    """Covariance between two input sets under a stationary kernel."""
    A = _matrix(A, "A")
    B = _matrix(B, "B")
    if A.shape[1] != B.shape[1]:
        raise Invalid("A and B must have the same number of input dimensions")
    if kernel not in KERNELS:
        raise Invalid("kernel must be one of: " + ", ".join(KERNELS))
    lengths = np.asarray(length_scale, dtype=float).reshape(-1)
    if lengths.size == 1:
        lengths = np.repeat(lengths, A.shape[1])
    if lengths.size != A.shape[1] or np.any(lengths <= 0) or not np.all(np.isfinite(lengths)):
        raise Invalid("length_scale must be one positive value, or one per input dimension")
    variance = _number(variance, "variance", low=1e-12)
    squares = _scaled_square_differences(A, B, lengths)
    K, _factor = _kernel_from_squares(squares, variance, kernel)
    return {
        "matrix": K,
        "kernel": kernel,
        "length_scale": [float(v) for v in lengths],
        "variance": variance,
        "scope": (
            "A stationary covariance assumption chosen by the caller. It encodes smoothness "
            "and a single length scale per dimension, which is a modelling choice, not a "
            "property established from the data."
        ),
    }


class TrainingSupport:
    """Where a learned component was actually trained, and how far a query is from it.

    The convex hull of the training inputs is the honest interpolation region for a
    smooth surrogate. `in_support` returns `interior`, `boundary` or `extrapolation`
    together with a signed Euclidean distance to the support boundary (negative inside).
    Above `max_hull_dimension`, or when the training points are degenerate, the hull is
    replaced by the axis-aligned bounding box, which is an OUTER approximation: it can
    call a point interior that the true hull excludes. The method used is reported.

    Being inside the hull is not sufficient for accuracy. A point can sit inside the hull
    in a region with almost no training data, so `local_density` and the distance to the
    nearest training point are reported alongside the status.
    """

    def __init__(self, X, boundary_fraction=0.05, max_hull_dimension=MAX_HULL_DIMENSION, neighbours=5):
        self.X = _matrix(X, "X", max_rows=MAX_SUPPORT_POINTS)
        self.n_samples, self.n_dimensions = self.X.shape
        self.boundary_fraction = _number(boundary_fraction, "boundary_fraction", low=0.0, high=0.5)
        self.neighbours = _integer(neighbours, "neighbours", 1, max(1, min(50, self.n_samples)))
        max_hull_dimension = _integer(max_hull_dimension, "max_hull_dimension", 1, MAX_HULL_DIMENSION)

        self.lower = self.X.min(axis=0)
        self.upper = self.X.max(axis=0)
        extent = self.upper - self.lower
        positive = extent[extent > 0]
        self.scale = float(positive.mean()) if positive.size else 1.0
        self.tolerance = self.boundary_fraction * self.scale
        self.hull_equations = None
        self.method = "bounding_box"
        self.hull_note = ""
        if self.n_dimensions == 1:
            self.method = "interval"
            self.hull_note = "In one dimension the bounding interval is the exact convex hull."
        elif self.n_dimensions <= max_hull_dimension and self.n_samples > self.n_dimensions:
            try:
                hull = ConvexHull(self.X)
                self.hull_equations = np.asarray(hull.equations, dtype=float)
                self.method = "convex_hull"
                self.hull_volume = float(hull.volume)
            except (QhullError, ValueError) as error:
                self.hull_note = (
                    "Convex hull unavailable (" + type(error).__name__ + "); the axis-aligned "
                    "bounding box is used instead and is an outer approximation."
                )
        else:
            self.hull_note = (
                f"Input dimension {self.n_dimensions} exceeds the hull limit "
                f"{max_hull_dimension}; the axis-aligned bounding box is used instead and is "
                "an outer approximation of the true support."
            )
        spacing = np.linalg.norm(extent) / max(1.0, self.n_samples ** (1.0 / max(1, self.n_dimensions)))
        self.typical_spacing = float(spacing) if math.isfinite(spacing) and spacing > 0 else 1.0

    def _box_distance(self, points):
        gaps = np.maximum(self.lower[None, :] - points, points - self.upper[None, :])
        outside = np.maximum(gaps, 0.0)
        exterior = np.linalg.norm(outside, axis=1)
        interior = gaps.max(axis=1)
        return np.where(exterior > 0, exterior, interior)

    def signed_distance(self, points):
        """Signed Euclidean distance to the support boundary; negative inside."""
        points = _matrix(points, "points", max_cols=self.n_dimensions)
        if points.shape[1] != self.n_dimensions:
            raise Invalid("Query points must match the training input dimension")
        if self.hull_equations is None:
            return self._box_distance(points)
        return (points @ self.hull_equations[:, :-1].T + self.hull_equations[:, -1]).max(axis=1)

    def _neighbour_distances(self, points):
        chunk = max(1, int(2_000_000 / max(1, self.n_samples)))
        nearest, densities = [], []
        for start in range(0, points.shape[0], chunk):
            block = points[start : start + chunk]
            distances = np.linalg.norm(block[:, None, :] - self.X[None, :, :], axis=2)
            ordered = np.sort(distances, axis=1)
            k = min(self.neighbours, ordered.shape[1])
            nearest.append(ordered[:, 0])
            densities.append(ordered[:, :k].mean(axis=1))
        return np.concatenate(nearest), np.concatenate(densities)

    def classify(self, points):
        """Classify a batch of query points against the recorded training support."""
        points = _matrix(points, "points", max_cols=self.n_dimensions)
        if points.shape[1] != self.n_dimensions:
            raise Invalid("Query points must match the training input dimension")
        distances = self.signed_distance(points)
        statuses = np.where(
            distances > self.tolerance,
            "extrapolation",
            np.where(distances < -self.tolerance, "interior", "boundary"),
        )
        nearest, mean_neighbour = self._neighbour_distances(points)
        sparse_interior = (statuses != "extrapolation") & (
            nearest > 3.0 * max(self.typical_spacing, 1e-12)
        )
        return {
            "status": [str(s) for s in statuses],
            "signed_distance": [float(v) for v in distances],
            "nearest_training_distance": [float(v) for v in nearest],
            "local_density_radius": [float(v) for v in mean_neighbour],
            "counts": {
                status: int(np.sum(statuses == status)) for status in SUPPORT_STATUSES
            },
            "n_extrapolating": int(np.sum(statuses == "extrapolation")),
            "sparse_interior_points": int(np.sum(sparse_interior)),
            "method": self.method,
            "tolerance": float(self.tolerance),
            "scope": (
                "Geometry of the training inputs only. Interior status says a query lies inside "
                "the region the component was fitted on; it does not certify accuracy there, and "
                "a sparsely sampled interior region can be as unreliable as an exterior one."
            ),
        }

    def in_support(self, x):
        """Classify one query point."""
        point = np.asarray(x, dtype=float).reshape(1, -1)
        result = self.classify(point)
        return {
            "status": result["status"][0],
            "signed_distance": result["signed_distance"][0],
            "distance_outside": max(0.0, result["signed_distance"][0]),
            "nearest_training_distance": result["nearest_training_distance"][0],
            "local_density_radius": result["local_density_radius"][0],
            "method": self.method,
            "tolerance": float(self.tolerance),
            "scope": result["scope"],
        }

    def summary(self):
        """Report the recorded support without querying it."""
        return {
            "method": self.method,
            "note": self.hull_note,
            "n_training_points": int(self.n_samples),
            "n_dimensions": int(self.n_dimensions),
            "lower": [float(v) for v in self.lower],
            "upper": [float(v) for v in self.upper],
            "tolerance": float(self.tolerance),
            "boundary_fraction": float(self.boundary_fraction),
            "typical_spacing": float(self.typical_spacing),
            "scope": (
                "A description of the training inputs. It is not a statement about the domain "
                "over which the underlying system is valid or measured."
            ),
        }


def _standardize(values):
    centre = values.mean(axis=0)
    spread = values.std(axis=0)
    spread = np.where(spread > 1e-12, spread, 1.0)
    return centre, spread


class GPEmulator:
    """Gaussian process regression with a marginal-likelihood fit and honest variance.

    Single scalar output. Inputs are standardized and the target is centred and scaled
    internally, so hyperparameter bounds are interpretable. Hyperparameters are the log
    signal variance, one log length scale per input dimension when `ard` is set, and the
    log nugget; they are optimized by L-BFGS-B on the negative log marginal likelihood
    with the analytic gradient implemented below.

    Training size is capped at MAX_GP_POINTS because the fit factorizes an n-by-n
    covariance matrix once per likelihood evaluation: O(n^3) time and O(n^2) memory.
    Subsampling to fit the cap is the caller's decision, not something done silently here.

    The posterior variance is correct under the assumed kernel, the assumed stationarity
    and the fitted nugget. It is not a calibrated error bar for a misspecified kernel and
    it is not a bound under extrapolation, which is why every prediction also carries a
    `TrainingSupport` verdict.
    """

    def __init__(
        self,
        kernel="matern52",
        ard=True,
        nugget=1e-6,
        optimize=True,
        n_restarts=2,
        max_iterations=200,
        seed=2026,
        boundary_fraction=0.05,
    ):
        if kernel not in KERNELS:
            raise Invalid("kernel must be one of: " + ", ".join(KERNELS))
        if not isinstance(ard, bool) or not isinstance(optimize, bool):
            raise Invalid("ard and optimize must be booleans")
        self.kernel = kernel
        self.ard = ard
        self.nugget = _number(nugget, "nugget", low=1e-12, high=1e3)
        self.optimize = optimize
        self.n_restarts = _integer(n_restarts, "n_restarts", 1, 16)
        self.max_iterations = _integer(max_iterations, "max_iterations", 1, MAX_OPTIMIZER_ITERATIONS)
        self.seed = _integer(seed, "seed", 0, 2**31 - 1)
        self.boundary_fraction = _number(boundary_fraction, "boundary_fraction", low=0.0, high=0.5)
        self.fitted = False
        self.support = None

    def _unpack(self, theta):
        variance = math.exp(theta[0])
        lengths = np.exp(theta[1 : 1 + self.n_length_scales])
        if lengths.size == 1:
            lengths = np.repeat(lengths, self.n_dimensions)
        nugget = math.exp(theta[-1])
        return variance, lengths, nugget

    def _negative_log_marginal_likelihood(self, theta):
        variance, lengths, nugget = self._unpack(theta)
        squares = _scaled_square_differences(self.Xs, self.Xs, lengths)
        Ks, factor = _kernel_from_squares(squares, variance, self.kernel)
        n = Ks.shape[0]
        K = Ks + (nugget + 1e-10) * np.eye(n)
        try:
            cholesky = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e12, np.zeros_like(theta)
        weights = cho_solve(cholesky, self.ys)
        log_determinant = 2.0 * np.sum(np.log(np.abs(np.diag(cholesky[0]))))
        nll = 0.5 * float(self.ys @ weights) + 0.5 * log_determinant + 0.5 * n * math.log(2 * math.pi)
        inverse = cho_solve(cholesky, np.eye(n))
        adjoint = 0.5 * (inverse - np.outer(weights, weights))
        gradient = np.zeros_like(theta)
        gradient[0] = float(np.sum(adjoint * Ks))
        if self.n_length_scales == 1:
            gradient[1] = float(np.sum(adjoint * (factor * squares.sum(axis=-1))))
        else:
            for dimension in range(self.n_dimensions):
                gradient[1 + dimension] = float(
                    np.sum(adjoint * (factor * squares[:, :, dimension]))
                )
        gradient[-1] = float(nugget * np.trace(adjoint))
        if not math.isfinite(nll):
            return 1e12, np.zeros_like(theta)
        return nll, gradient

    def fit(self, X, y, fixed_noise_variance=None):
        """Fit hyperparameters by marginal likelihood and record the training support.

        `fixed_noise_variance`, in the units of `y`, holds the nugget at a known observation
        variance instead of letting the likelihood choose it. Use it whenever the measurement
        error is actually known: a free nugget and a short length scale explain the same data,
        so fixing what is known is what makes the rest of the fit mean anything.
        """
        X = _matrix(X, "X", max_rows=MAX_GP_POINTS)
        target = np.asarray(y, dtype=float).reshape(-1)
        if target.size != X.shape[0] or not np.all(np.isfinite(target)):
            raise Invalid("y must hold one finite value per training input")
        if X.shape[0] < 2:
            raise Invalid("A Gaussian process fit needs at least 2 training points")
        self.X_raw = X
        self.n_dimensions = X.shape[1]
        self.n_length_scales = self.n_dimensions if self.ard else 1
        self.x_centre, self.x_spread = _standardize(X)
        self.Xs = (X - self.x_centre) / self.x_spread
        self.y_centre = float(target.mean())
        self.y_spread = float(target.std()) if target.std() > 1e-12 else 1.0
        self.ys = (target - self.y_centre) / self.y_spread

        nugget_start = self.nugget
        nugget_bound = (math.log(1e-10), math.log(10.0))
        nugget_fixed = fixed_noise_variance is not None
        if nugget_fixed:
            scaled = _number(fixed_noise_variance, "fixed_noise_variance", low=0.0)
            nugget_start = max(scaled / (self.y_spread**2), 1e-10)
            nugget_bound = (math.log(nugget_start), math.log(nugget_start))
        bounds = (
            [(-6.0, 6.0)]
            + [(-5.0, 5.0)] * self.n_length_scales
            + [nugget_bound]
        )
        start = np.array(
            [0.0] + [0.0] * self.n_length_scales + [math.log(nugget_start)], dtype=float
        )
        generator = np.random.default_rng(self.seed)
        starts = [start]
        for _ in range(self.n_restarts - 1):
            jitter = generator.uniform(-1.0, 1.0, size=start.size)
            starts.append(np.clip(start + jitter, [b[0] for b in bounds], [b[1] for b in bounds]))

        best = None
        evaluations, iterations, budget_hit = 0, 0, False
        if self.optimize:
            for candidate in starts:
                result = minimize(
                    self._negative_log_marginal_likelihood,
                    candidate,
                    jac=True,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": self.max_iterations},
                )
                evaluations += int(result.nfev)
                iterations += int(result.nit)
                budget_hit = budget_hit or int(result.nit) >= self.max_iterations
                if best is None or result.fun < best.fun:
                    best = result
            theta = np.asarray(best.x, dtype=float)
            converged = bool(best.success)
            message = str(best.message)
        else:
            theta = start
            converged, message = True, "Hyperparameter optimization disabled by the caller."

        self.theta = theta
        variance, lengths, nugget = self._unpack(theta)
        squares = _scaled_square_differences(self.Xs, self.Xs, lengths)
        Ks, _factor = _kernel_from_squares(squares, variance, self.kernel)
        K = Ks + (nugget + 1e-10) * np.eye(Ks.shape[0])
        try:
            self._cholesky = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            raise Invalid("Covariance matrix is not positive definite at the fitted hyperparameters") from None
        self._weights = cho_solve(self._cholesky, self.ys)
        self._lengths = lengths
        self._variance = variance
        self._nugget = nugget
        self.support = TrainingSupport(X, boundary_fraction=self.boundary_fraction)
        self.fitted = True
        nll, _gradient = self._negative_log_marginal_likelihood(theta)
        residual = self.mean(X) - target
        warnings = []
        if budget_hit:
            warnings.append(
                f"L-BFGS-B reached its {self.max_iterations}-iteration budget; the reported "
                "hyperparameters are the last iterate, not a converged optimum."
            )
        if not converged:
            warnings.append("Marginal-likelihood optimization reported failure: " + message)
        if nugget > 0.5 and not nugget_fixed:
            warnings.append(
                "The fitted nugget absorbs most of the target variance; the mean function is "
                "close to a constant and the data may carry little learnable signal."
            )
        return {
            "kernel": self.kernel,
            "ard": self.ard,
            "n_train": int(X.shape[0]),
            "n_dimensions": int(self.n_dimensions),
            "signal_variance": float(variance * self.y_spread**2),
            "length_scale_standardized": [float(v) for v in lengths],
            "length_scale_input_units": [float(v * s) for v, s in zip(lengths, self.x_spread)],
            "nugget": float(nugget * self.y_spread**2),
            "nugget_fixed": bool(nugget_fixed),
            "log_marginal_likelihood": float(-nll),
            "converged": bool(converged and not budget_hit),
            "optimizer_message": message,
            "iterations": int(iterations),
            "function_evaluations": int(evaluations),
            "train_rmse": float(np.sqrt(np.mean(residual**2))),
            "support": self.support.summary(),
            "warnings": warnings,
            "scope": SCOPE
            + " The marginal likelihood ranks hyperparameters under the assumed kernel; it does "
            "not test whether that kernel is the right assumption.",
        }

    def _cross_covariance(self, X):
        Xs = (X - self.x_centre) / self.x_spread
        squares = _scaled_square_differences(Xs, self.Xs, self._lengths)
        Ks, _factor = _kernel_from_squares(squares, self._variance, self.kernel)
        return Ks

    def mean(self, X):
        """Posterior mean only, without the support report (fast path for inner loops)."""
        if not self.fitted:
            raise Invalid("GPEmulator.fit must be called before prediction")
        X = _matrix(X, "X", max_cols=self.n_dimensions)
        if X.shape[1] != self.n_dimensions:
            raise Invalid("Prediction inputs must match the training input dimension")
        return self._cross_covariance(X) @ self._weights * self.y_spread + self.y_centre

    def predict(self, X, include_observation_noise=False):
        """Posterior mean and variance, with the training-support verdict for every point."""
        if not self.fitted:
            raise Invalid("GPEmulator.fit must be called before prediction")
        X = _matrix(X, "X", max_cols=self.n_dimensions)
        if X.shape[1] != self.n_dimensions:
            raise Invalid("Prediction inputs must match the training input dimension")
        Ks = self._cross_covariance(X)
        mean = Ks @ self._weights * self.y_spread + self.y_centre
        solved = cho_solve(self._cholesky, Ks.T)
        prior = self._variance
        variance = prior - np.einsum("ij,ji->i", Ks, solved)
        variance = np.maximum(variance, 0.0)
        if include_observation_noise:
            variance = variance + self._nugget
        variance = variance * self.y_spread**2
        support = self.support.classify(X)
        return {
            "mean": mean,
            "variance": variance,
            "standard_deviation": np.sqrt(variance),
            "support_status": support["status"],
            "signed_distance_to_support": support["signed_distance"],
            "n_extrapolating": support["n_extrapolating"],
            "includes_observation_noise": bool(include_observation_noise),
            "prior_variance": float(prior * self.y_spread**2),
            "scope": (
                "Posterior moments under the assumed kernel and fitted hyperparameters. At "
                "extrapolation points the variance reverts towards the prior, which reflects the "
                "kernel's assumption rather than measured error; treat those predictions as "
                "unsupported regardless of the interval they carry."
            ),
        }


class NeuralSurrogate:
    """A small numpy MLP with analytic backpropagation, fitted by L-BFGS-B.

    The network is a plain feed-forward stack with tanh or ReLU hidden activations and a
    linear output. Weights are flattened into one vector so `scipy.optimize.minimize` can
    run L-BFGS-B over them; the gradient handed to the optimizer is the real
    backpropagation gradient computed in `_loss_and_gradient`, not a finite-difference
    approximation of the whole weight vector. `gradient_check` exposes the comparison so
    that claim is testable rather than asserted.

    Parameter count is capped: this is a surrogate for a component inside a mechanistic
    model, not a general-purpose network, and a full-batch quasi-Newton fit over many
    thousands of weights is neither fast nor well conditioned.
    """

    def __init__(
        self,
        hidden=(16,),
        activation="tanh",
        weight_decay=1e-6,
        seed=2026,
        max_iterations=400,
        boundary_fraction=0.05,
    ):
        if activation not in ACTIVATIONS:
            raise Invalid("activation must be one of: " + ", ".join(ACTIVATIONS))
        hidden = tuple(hidden)
        if not hidden or len(hidden) > MAX_HIDDEN_LAYERS:
            raise Invalid(f"hidden must name 1 to {MAX_HIDDEN_LAYERS} layers")
        for width in hidden:
            _integer(width, "hidden layer width", 1, MAX_HIDDEN_UNITS)
        self.hidden = tuple(int(w) for w in hidden)
        self.activation = activation
        self.weight_decay = _number(weight_decay, "weight_decay", low=0.0, high=1e3)
        self.seed = _integer(seed, "seed", 0, 2**31 - 1)
        self.max_iterations = _integer(max_iterations, "max_iterations", 1, MAX_OPTIMIZER_ITERATIONS)
        self.boundary_fraction = _number(boundary_fraction, "boundary_fraction", low=0.0, high=0.5)
        self.fitted = False
        self.support = None

    def _configure(self, X, y):
        X = _matrix(X, "X")
        Y = _matrix(y, "y", max_cols=MAX_OUTPUT_DIMENSION)
        if Y.shape[0] != X.shape[0]:
            raise Invalid("X and y must have the same number of rows")
        self.n_dimensions = X.shape[1]
        self.n_outputs = Y.shape[1]
        self.shapes = []
        widths = [self.n_dimensions, *self.hidden, self.n_outputs]
        total = 0
        for index in range(len(widths) - 1):
            self.shapes.append((widths[index], widths[index + 1]))
            total += widths[index] * widths[index + 1] + widths[index + 1]
        if total > MAX_MLP_PARAMETERS:
            raise Invalid(
                f"Network has {total} parameters, above the {MAX_MLP_PARAMETERS} limit for a "
                "full-batch quasi-Newton fit"
            )
        self.n_parameters = total
        self.x_centre, self.x_spread = _standardize(X)
        self.y_centre, self.y_spread = _standardize(Y)
        return (X - self.x_centre) / self.x_spread, (Y - self.y_centre) / self.y_spread, X

    def initial_weights(self, seed=None):
        """Deterministic weight initialization for a configured network."""
        generator = np.random.default_rng(self.seed if seed is None else _integer(seed, "seed", 0, 2**31 - 1))
        pieces = []
        for fan_in, fan_out in self.shapes:
            gain = 2.0 if self.activation == "relu" else 1.0
            scale = math.sqrt(gain / max(1, fan_in))
            pieces.append(generator.normal(0.0, scale, size=fan_in * fan_out))
            pieces.append(np.zeros(fan_out))
        return np.concatenate(pieces)

    def _unpack(self, flat):
        layers, offset = [], 0
        for fan_in, fan_out in self.shapes:
            weight = flat[offset : offset + fan_in * fan_out].reshape(fan_in, fan_out)
            offset += fan_in * fan_out
            bias = flat[offset : offset + fan_out]
            offset += fan_out
            layers.append((weight, bias))
        return layers

    def _activate(self, z):
        return np.tanh(z) if self.activation == "tanh" else np.maximum(z, 0.0)

    def _activation_gradient(self, z, a):
        return 1.0 - a * a if self.activation == "tanh" else (z > 0.0).astype(float)

    def _propagate(self, layers, Xs):
        activations, pre_activations = [Xs], []
        current = Xs
        for index, (weight, bias) in enumerate(layers):
            z = current @ weight + bias
            pre_activations.append(z)
            current = z if index == len(layers) - 1 else self._activate(z)
            activations.append(current)
        return activations, pre_activations

    def _loss_and_gradient(self, flat, Xs, Ys):
        layers = self._unpack(flat)
        activations, pre_activations = self._propagate(layers, Xs)
        output = activations[-1]
        difference = output - Ys
        count = difference.size
        loss = float(np.sum(difference * difference) / count)
        decay = self.weight_decay
        if decay:
            loss += decay * float(sum(np.sum(w * w) for w, _b in layers))
        delta = 2.0 * difference / count
        gradients = [None] * len(layers)
        for index in range(len(layers) - 1, -1, -1):
            weight, _bias = layers[index]
            grad_weight = activations[index].T @ delta
            if decay:
                grad_weight = grad_weight + 2.0 * decay * weight
            grad_bias = delta.sum(axis=0)
            gradients[index] = (grad_weight, grad_bias)
            if index > 0:
                delta = (delta @ weight.T) * self._activation_gradient(
                    pre_activations[index - 1], activations[index]
                )
        flat_gradient = np.concatenate(
            [piece.reshape(-1) for pair in gradients for piece in pair]
        )
        return loss, flat_gradient

    def fit(self, X, y, initial_weights=None):
        """Train the network and record its training support."""
        Xs, Ys, X_raw = self._configure(X, y)
        start = (
            self.initial_weights()
            if initial_weights is None
            else np.asarray(initial_weights, dtype=float).reshape(-1)
        )
        if start.size != self.n_parameters:
            raise Invalid("initial_weights does not match the configured network size")
        result = minimize(
            self._loss_and_gradient,
            start,
            args=(Xs, Ys),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iterations},
        )
        self.weights = np.asarray(result.x, dtype=float)
        self.fitted = True
        self.support = TrainingSupport(X_raw, boundary_fraction=self.boundary_fraction)
        budget_hit = int(result.nit) >= self.max_iterations
        residual = self.forward(X_raw) - _matrix(y, "y", max_cols=MAX_OUTPUT_DIMENSION)
        warnings = []
        if budget_hit:
            warnings.append(
                f"L-BFGS-B reached its {self.max_iterations}-iteration budget; training stopped "
                "on the budget, not on a convergence criterion."
            )
        if not bool(result.success):
            warnings.append("Optimizer reported failure: " + str(result.message))
        return {
            "converged": bool(result.success) and not budget_hit,
            "optimizer_message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "final_objective": float(result.fun),
            "gradient_norm": float(np.linalg.norm(result.jac)) if result.jac is not None else None,
            "train_rmse": float(np.sqrt(np.mean(residual**2))),
            "n_parameters": int(self.n_parameters),
            "architecture": [self.n_dimensions, *self.hidden, self.n_outputs],
            "activation": self.activation,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "support": self.support.summary(),
            "warnings": warnings,
            "scope": SCOPE
            + " A converged optimizer reports a stationary point of this objective on this "
            "training set; it says nothing about generalization.",
        }

    def forward(self, X):
        """Network output in the original target units (fast path, no support report)."""
        if not self.fitted:
            raise Invalid("NeuralSurrogate.fit must be called before prediction")
        X = _matrix(X, "X", max_cols=self.n_dimensions)
        if X.shape[1] != self.n_dimensions:
            raise Invalid("Prediction inputs must match the training input dimension")
        Xs = (X - self.x_centre) / self.x_spread
        activations, _pre = self._propagate(self._unpack(self.weights), Xs)
        return activations[-1] * self.y_spread + self.y_centre

    def forward_with_weights(self, X, weights):
        """Network output for a candidate weight vector, in original target units.

        Used by callers that optimize the weights outside `fit` - a Universal Differential
        Equation fit, for instance - so the standardization fixed at `fit` time is reused
        rather than re-derived. Returns a bare array; the support report is not attached.
        """
        if not self.shapes:
            raise Invalid("NeuralSurrogate must be configured by fit before evaluation")
        X = _matrix(X, "X", max_cols=self.n_dimensions)
        if X.shape[1] != self.n_dimensions:
            raise Invalid("Prediction inputs must match the training input dimension")
        flat = np.asarray(weights, dtype=float).reshape(-1)
        if flat.size != self.n_parameters:
            raise Invalid("weights does not match the configured network size")
        Xs = (X - self.x_centre) / self.x_spread
        activations, _pre = self._propagate(self._unpack(flat), Xs)
        return activations[-1] * self.y_spread + self.y_centre

    def predict(self, X):
        """Network output with the training-support verdict for every point."""
        X = _matrix(X, "X", max_cols=self.n_dimensions)
        values = self.forward(X)
        support = self.support.classify(X)
        return {
            "mean": values,
            "support_status": support["status"],
            "signed_distance_to_support": support["signed_distance"],
            "n_extrapolating": support["n_extrapolating"],
            "nearest_training_distance": support["nearest_training_distance"],
            "scope": (
                "A deterministic function fitted to the training samples. It carries no "
                "uncertainty estimate of its own; outside the reported support its output is an "
                "unconstrained extrapolation of the fitted weights."
            ),
        }

    def gradient_check(self, X, y, epsilon=1e-6, tolerance=1e-6, max_parameters=200, seed=None):
        """Compare the analytic backpropagation gradient with central finite differences.

        Runs on the configured architecture at a deterministic initial weight vector. A
        large discrepancy means the training gradient is wrong, which no amount of
        optimizer tuning will fix.
        """
        Xs, Ys, _X_raw = self._configure(X, y)
        epsilon = _number(epsilon, "epsilon", low=1e-10, high=1e-2)
        tolerance = _number(tolerance, "tolerance", low=0.0)
        max_parameters = _integer(max_parameters, "max_parameters", 1, MAX_MLP_PARAMETERS)
        weights = self.initial_weights(seed=seed)
        _loss, analytic = self._loss_and_gradient(weights, Xs, Ys)
        indices = np.arange(min(self.n_parameters, max_parameters))
        numeric = np.zeros(indices.size)
        for position, index in enumerate(indices):
            shifted = weights.copy()
            shifted[index] += epsilon
            upper, _g = self._loss_and_gradient(shifted, Xs, Ys)
            shifted[index] -= 2.0 * epsilon
            lower, _g = self._loss_and_gradient(shifted, Xs, Ys)
            numeric[position] = (upper - lower) / (2.0 * epsilon)
        analytic_subset = analytic[indices]
        absolute = np.abs(analytic_subset - numeric)
        denominator = np.maximum(np.abs(analytic_subset) + np.abs(numeric), 1e-12)
        relative = absolute / denominator
        return {
            "max_absolute_difference": float(absolute.max()) if absolute.size else 0.0,
            "max_relative_difference": float(relative.max()) if relative.size else 0.0,
            "n_parameters_checked": int(indices.size),
            "n_parameters": int(self.n_parameters),
            "epsilon": epsilon,
            "tolerance": tolerance,
            "passed": bool(relative.max() <= tolerance) if relative.size else True,
            "analytic_gradient_norm": float(np.linalg.norm(analytic)),
            "scope": (
                "Checks that the implemented gradient matches the implemented loss. It does not "
                "check that the loss, the architecture or the data are appropriate."
            ),
        }


def approximation_error(predictions, targets, inputs=None, support=None):
    """Held-out error, reported separately inside and outside the training support.

    An aggregate RMSE over a test set that straddles the training hull is misleading: the
    interior and exterior numbers are usually different by an order of magnitude, and it
    is the exterior number that governs what happens when the surrogate is coupled to a
    mechanistic model driven into a new regime. Supply `inputs` and a `TrainingSupport`
    to get the split; without them only the aggregate is reported, and it is labelled as
    such.
    """
    predicted = np.asarray(predictions, dtype=float).reshape(-1)
    observed = np.asarray(targets, dtype=float).reshape(-1)
    if predicted.size != observed.size or predicted.size == 0:
        raise Invalid("predictions and targets must be nonempty and the same length")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(observed)):
        raise Invalid("predictions and targets must be finite")
    residual = predicted - observed
    centred = observed - observed.mean()
    ss_tot = float(centred @ centred)
    report = {
        "n": int(residual.size),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "max_absolute_error": float(np.max(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "r_squared": None if ss_tot <= 0 else float(1.0 - float(residual @ residual) / ss_tot),
        "by_support": None,
        "extrapolation_error_ratio": None,
        "support_method": None,
    }
    if inputs is not None and support is not None:
        if not isinstance(support, TrainingSupport):
            raise Invalid("support must be a TrainingSupport instance")
        points = _matrix(inputs, "inputs", max_cols=support.n_dimensions)
        if points.shape[0] != residual.size:
            raise Invalid("inputs must hold one row per prediction")
        classified = support.classify(points)
        statuses = np.asarray(classified["status"])
        grouped = {}
        for status in SUPPORT_STATUSES:
            mask = statuses == status
            if not mask.any():
                grouped[status] = {"n": 0, "rmse": None, "max_absolute_error": None, "bias": None}
                continue
            block = residual[mask]
            grouped[status] = {
                "n": int(mask.sum()),
                "rmse": float(np.sqrt(np.mean(block**2))),
                "max_absolute_error": float(np.max(np.abs(block))),
                "bias": float(np.mean(block)),
            }
        report["by_support"] = grouped
        report["support_method"] = support.method
        inside = grouped["interior"]["rmse"]
        outside = grouped["extrapolation"]["rmse"]
        if inside and outside:
            report["extrapolation_error_ratio"] = float(outside / inside)
    report["scope"] = (
        "Empirical error on the supplied held-out samples only. It generalizes to new inputs "
        "only insofar as they are drawn like these; the extrapolation figure is a sample from "
        "one direction of departure, not an upper bound on extrapolation error."
    )
    return report
