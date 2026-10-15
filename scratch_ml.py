"""
scratch_ml.py
=============
Customer Purchase Prediction - machine learning built from scratch.

Everything in this file uses only NumPy / pandas (no scikit-learn):

  * data cleaning + feature preprocessing (log transform, standardisation, one-hot)
  * stratified train/test split and stratified K-fold
  * evaluation metrics (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)
  * Logistic Regression  (Newton-Raphson / IRLS, L2 regularised)
  * Decision Tree        (CART, Gini / Entropy, pre-pruning)
  * K-Nearest Neighbours (vectorised Euclidean distance, uniform / distance vote)
  * Support Vector Machine (hinge loss + L2, RBF kernel via Random Fourier Features,
                            Platt scaling for probabilities)
  * cross-validated hyper-parameter search with decision-threshold tuning

The file is written in "percent" format (# %% markers) so that the notebook can
show the exact same code cell by cell.
"""

# %% imports
import numpy as np
import pandas as pd
from itertools import product

SEED = 42
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
NUMERIC_COLS = ["Administrative", "Informational", "ProductRelated", "ProductRelated_Duration",
                "BounceRates", "ExitRates", "PageValues", "SpecialDay"]
LOG_COLS = ["Administrative", "Informational", "ProductRelated", "ProductRelated_Duration", "PageValues"]
RAW_FEATURES = NUMERIC_COLS + ["Month", "VisitorType", "Weekend"]
TARGET = "Revenue"


# %% data_cleaning
def load_and_clean(path):
    """Read the CSV, drop rows with missing values and exact duplicate sessions."""
    raw = pd.read_csv(path)
    n_raw = len(raw)
    df = raw.dropna()
    n_missing = n_raw - len(df)
    df = df.drop_duplicates()
    n_dup = n_raw - n_missing - len(df)
    df = df.reset_index(drop=True)
    info = {"rows_raw": n_raw, "rows_missing": n_missing,
            "rows_duplicate": n_dup, "rows_clean": len(df)}
    return df, info


# %% preprocessing
class Preprocessor:
    """
    Raw session record -> numeric feature vector.

    * heavily right-skewed count / duration columns get log(1 + x)
    * continuous columns are standardised with TRAIN mean / std only (no leakage)
    * Month and VisitorType are one-hot encoded (label encoding would impose a
      fake order, which hurts distance based models such as KNN / SVM / LR)
    * Weekend is a 0/1 flag
    """

    def fit(self, df):
        self.months_ = [m for m in MONTH_ORDER if m in set(df["Month"].astype(str))]
        self.visitors_ = sorted(df["VisitorType"].astype(str).unique())
        cont = self._continuous(df)
        self.mean_ = cont.mean(axis=0)
        self.std_ = cont.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def _continuous(self, df):
        cont = df[NUMERIC_COLS].astype(float).copy()
        for c in LOG_COLS:
            cont[c] = np.log1p(cont[c])
        return cont.to_numpy()

    def transform(self, df):
        cont = (self._continuous(df) - self.mean_) / self.std_
        weekend = df["Weekend"].astype(int).to_numpy().reshape(-1, 1).astype(float)
        month = df["Month"].astype(str).to_numpy()
        visitor = df["VisitorType"].astype(str).to_numpy()
        month_oh = np.stack([(month == m).astype(float) for m in self.months_], axis=1)
        visitor_oh = np.stack([(visitor == v).astype(float) for v in self.visitors_], axis=1)
        return np.hstack([cont, weekend, month_oh, visitor_oh])

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    @property
    def feature_names_(self):
        return (NUMERIC_COLS + ["Weekend"]
                + ["Month_" + m for m in self.months_]
                + ["Visitor_" + v for v in self.visitors_])


# %% splitting
def stratified_split(y, test_size=0.20, seed=SEED):
    """Stratified train/test split (keeps the 15.5 % purchase rate in both parts)."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return rng.permutation(train_idx), rng.permutation(test_idx)


def stratified_kfold(y, n_splits=5, seed=SEED):
    """Yield (train_idx, val_idx) pairs, each fold keeping the class ratio."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    folds = [[] for _ in range(n_splits)]
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        for k, part in enumerate(np.array_split(idx, n_splits)):
            folds[k].extend(part)
    for k in range(n_splits):
        val = np.array(folds[k])
        train = np.array([i for j in range(n_splits) if j != k for i in folds[j]])
        yield train, val


# %% metrics
def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return np.array([[tn, fp], [fn, tp]])


def accuracy_score(y_true, y_pred):
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def precision_score(y_true, y_pred):
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    return float(tp / (tp + fp)) if (tp + fp) else 0.0


def recall_score(y_true, y_pred):
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    return float(tp / (tp + fn)) if (tp + fn) else 0.0


def f1_score(y_true, y_pred):
    p, r = precision_score(y_true, y_pred), recall_score(y_true, y_pred)
    return float(2 * p * r / (p + r)) if (p + r) else 0.0


def roc_curve(y_true, scores):
    """False-positive-rate / true-positive-rate pairs for every distinct score."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    tps, fps = np.cumsum(y), np.cumsum(1 - y)
    last = np.r_[np.where(np.diff(s))[0], len(y) - 1]
    tpr = np.r_[0.0, tps[last] / tps[-1]]
    fpr = np.r_[0.0, fps[last] / fps[-1]]
    return fpr, tpr


def roc_auc_score(y_true, scores):
    fpr, tpr = roc_curve(y_true, scores)
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


def best_f1_threshold(y_true, prob, grid=None):
    """Probability cut-off (searched on a 0.02..0.98 grid) that maximises F1."""
    grid = np.linspace(0.02, 0.98, 193) if grid is None else grid
    y = np.asarray(y_true).astype(bool)
    pred = np.asarray(prob)[:, None] >= grid[None, :]
    tp = (pred & y[:, None]).sum(axis=0)
    fp = (pred & ~y[:, None]).sum(axis=0)
    fn = y.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    best = np.flatnonzero(f1 >= f1.max() - 1e-12)
    j = best[len(best) // 2]                     # middle of the plateau = stable choice
    return float(grid[j]), float(f1[j])


def evaluate(y_true, prob, threshold=0.5):
    """All metrics for one model at a given decision threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0]
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "ROC-AUC": roc_auc_score(y_true, prob),
        "Confusion_Matrix": cm.tolist(),
        "Threshold": float(threshold),
    }


# %% base_classifier
def sigmoid(z):
    """Numerically stable logistic function."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def platt_apply(scores, A, B):
    """Platt scaling: raw SVM margin -> probability."""
    return sigmoid(-(A * np.asarray(scores, dtype=float) + B))


def fit_platt(scores, y, max_iter=100):
    """Fit p = 1 / (1 + exp(A*s + B)) by Newton's method (Platt, 1999)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y).astype(int)
    n_pos, n_neg = y.sum(), (1 - y).sum()
    t = np.where(y == 1, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
    A, B = 0.0, float(np.log((n_neg + 1.0) / (n_pos + 1.0)))

    def nll(a, b):
        z = a * s + b
        return float(np.sum(t * np.logaddexp(0, z) + (1 - t) * np.logaddexp(0, -z)))

    cur = nll(A, B)
    for _ in range(max_iter):
        p = platt_apply(s, A, B)
        g = np.array([np.sum((t - p) * s), np.sum(t - p)])
        w = p * (1 - p)
        H = np.array([[np.sum(s * s * w), np.sum(s * w)], [np.sum(s * w), np.sum(w)]]) + 1e-10 * np.eye(2)
        step = np.linalg.solve(H, g)
        scale = 1.0
        while scale > 1e-8:                       # backtracking line search
            new = nll(A - scale * step[0], B - scale * step[1])
            if new <= cur + 1e-12:
                break
            scale /= 2
        A, B = A - scale * step[0], B - scale * step[1]
        if abs(cur - new) < 1e-10:
            break
        cur = new
    return float(A), float(B)


class ClassifierMixin:
    """Shared predict / predict_proba logic. Subclasses implement fit() and score()."""
    threshold_ = 0.5
    platt_ = None

    def predict_proba(self, X):
        s = self.score(X)
        p = s if self.platt_ is None else platt_apply(s, *self.platt_)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold_).astype(int)


# %% logistic_regression
class LogisticRegressionScratch(ClassifierMixin):

    def __init__(self, l2=1.0, max_iter=50, tol=1e-8):
        self.l2, self.max_iter, self.tol = l2, max_iter, tol

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Xb = np.hstack([np.ones((len(X), 1)), X])
        reg = self.l2 * np.eye(Xb.shape[1])
        reg[0, 0] = 0.0
        w = np.zeros(Xb.shape[1])
        for it in range(self.max_iter):
            p = sigmoid(Xb @ w)
            grad = Xb.T @ (p - y) + reg @ w
            hess = Xb.T @ (Xb * (p * (1 - p))[:, None]) + reg + 1e-10 * np.eye(len(w))
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.max(np.abs(step)) < self.tol:
                break
        self.n_iter_ = it + 1
        self.bias_, self.coef_ = w[0], w[1:]
        return self

    def score(self, X):
        return sigmoid(np.asarray(X, dtype=float) @ self.coef_ + self.bias_)


# %% decision_tree
class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "proba", "n_samples")

    def __init__(self):
        self.feature, self.threshold, self.left, self.right = -1, 0.0, None, None
        self.proba, self.n_samples = 0.0, 0


class DecisionTreeScratch(ClassifierMixin):
    """
    CART binary classification tree.

    At every node all features and all mid-point thresholds are scanned (vectorised with
    sorted cumulative class weights) and the split with the largest impurity decrease
    (Gini or Entropy) is taken. Growth is limited by max_depth, min_samples_split and
    min_samples_leaf (pre-pruning). Leaf probability = fraction of purchases in the leaf.
    """

    def __init__(self, criterion="gini", max_depth=5, min_samples_split=10,
                 min_samples_leaf=5, class_weight=None):
        self.criterion, self.max_depth = criterion, max_depth
        self.min_samples_split, self.min_samples_leaf = min_samples_split, min_samples_leaf
        self.class_weight = class_weight

    def _impurity(self, w0, w1):
        tot = np.maximum(w0 + w1, 1e-12)
        p0, p1 = w0 / tot, w1 / tot
        if self.criterion == "gini":
            return 1.0 - p0 ** 2 - p1 ** 2
        def xlog(p):
            return np.where(p > 0, p * np.log2(np.where(p > 0, p, 1.0)), 0.0)
        return -(xlog(p0) + xlog(p1))

    def _best_split(self, X, y, sw):
        n, d = X.shape
        w1, w0 = sw * y, sw * (1 - y)
        W1, W0 = w1.sum(), w0.sum()
        parent = self._impurity(W0, W1) * (W0 + W1)
        best, best_gain = None, 1e-12
        left_n = np.arange(1, n)
        for j in range(d):
            order = np.argsort(X[:, j], kind="mergesort")
            xs = X[order, j]
            if xs[0] == xs[-1]:
                continue
            l1, l0 = np.cumsum(w1[order])[:-1], np.cumsum(w0[order])[:-1]
            r1, r0 = W1 - l1, W0 - l0
            valid = ((xs[1:] > xs[:-1]) & (left_n >= self.min_samples_leaf)
                     & (n - left_n >= self.min_samples_leaf))
            if not valid.any():
                continue
            child = self._impurity(l0, l1) * (l0 + l1) + self._impurity(r0, r1) * (r0 + r1)
            gain = np.where(valid, parent - child, -np.inf)
            i = int(np.argmax(gain))
            if gain[i] > best_gain:
                best_gain, best = gain[i], (j, (xs[i] + xs[i + 1]) / 2.0)
        return best, best_gain

    def _build(self, X, y, sw, depth):
        node = _Node()
        node.n_samples, node.proba = len(y), float(y.mean())
        too_deep = self.max_depth is not None and depth >= self.max_depth
        if too_deep or len(y) < self.min_samples_split or y.min() == y.max():
            return node
        split, gain = self._best_split(X, y, sw)
        if split is None:
            return node
        j, t = split
        mask = X[:, j] <= t
        node.feature, node.threshold = j, t
        self.importances_[j] += gain
        node.left = self._build(X[mask], y[mask], sw[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], sw[~mask], depth + 1)
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(float)
        sw = np.ones(len(y))
        if self.class_weight == "balanced":
            for c in (0.0, 1.0):
                sw[y == c] = len(y) / (2.0 * (y == c).sum())
        self.importances_ = np.zeros(X.shape[1])
        self.root_ = self._build(X, y, sw, 0)
        total = self.importances_.sum()
        if total > 0:
            self.importances_ = self.importances_ / total
        return self

    def _walk(self, node, X, idx, out):
        if node.left is None:
            out[idx] = node.proba
            return
        go_left = X[idx, node.feature] <= node.threshold
        self._walk(node.left, X, idx[go_left], out)
        self._walk(node.right, X, idx[~go_left], out)

    def score(self, X):
        X = np.asarray(X, dtype=float)
        out = np.zeros(len(X))
        self._walk(self.root_, X, np.arange(len(X)), out)
        return out

    def get_depth(self):
        def depth(n):
            return 0 if n.left is None else 1 + max(depth(n.left), depth(n.right))
        return depth(self.root_)

    def get_n_leaves(self):
        def leaves(n):
            return 1 if n.left is None else leaves(n.left) + leaves(n.right)
        return leaves(self.root_)


# %% knn
class KNNScratch(ClassifierMixin):
    """
    K-Nearest Neighbours. Euclidean distances are computed in vectorised chunks using
    ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b ; the k closest training points vote
    (uniformly, or weighted by 1/distance).
    """

    def __init__(self, k=15, weights="uniform", chunk=512):
        self.k, self.weights, self.chunk = k, weights, chunk

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y, dtype=float)
        self.sq_ = (self.X_ ** 2).sum(axis=1)
        return self

    def score(self, X):
        X = np.asarray(X, dtype=float)
        k = min(self.k, len(self.X_))
        out = np.zeros(len(X))
        for s in range(0, len(X), self.chunk):
            xb = X[s:s + self.chunk]
            d2 = (xb ** 2).sum(axis=1)[:, None] + self.sq_[None, :] - 2.0 * xb @ self.X_.T
            np.maximum(d2, 0.0, out=d2)
            nn = np.argpartition(d2, k - 1, axis=1)[:, :k]
            labels = self.y_[nn]
            if self.weights == "distance":
                w = 1.0 / (np.sqrt(np.take_along_axis(d2, nn, axis=1)) + 1e-9)
                out[s:s + self.chunk] = (w * labels).sum(axis=1) / w.sum(axis=1)
            else:
                out[s:s + self.chunk] = labels.mean(axis=1)
        return out


# %% svm
class SVMScratch(ClassifierMixin):
    """
    Soft-margin SVM trained directly on the primal objective

        min_{w,b}  1/(2 C n) ||w||^2  +  (1/n) sum_i  s_i * max(0, 1 - y_i (w.z_i + b))

    with full-batch sub-gradient descent (Adam step sizes).  For kernel='rbf' the inputs are
    first mapped with Random Fourier Features (Rahimi & Recht, 2007), z(x) = sqrt(2/D) cos(Wx+b),
    which satisfy z(x).z(x') ~ exp(-gamma ||x-x'||^2); the SVM is then linear in z-space.
    kernel='linear' trains on the raw features.  Probabilities come from Platt scaling.
    """

    def __init__(self, C=1.0, kernel="rbf", gamma=0.05, n_components=500, n_iter=300,
                 lr=0.05, class_weight=None, seed=SEED):
        self.C, self.kernel, self.gamma = C, kernel, gamma
        self.n_components, self.n_iter, self.lr = n_components, n_iter, lr
        self.class_weight, self.seed = class_weight, seed
        self.platt_ = (-1.0, 0.0)               # identity-like default (p = sigmoid(margin))

    def _map(self, X):
        X = np.asarray(X, dtype=float)
        if self.kernel == "linear":
            return X
        return np.sqrt(2.0 / self.n_components) * np.cos(X @ self.W_ + self.b_)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(float)
        ys = 2.0 * y - 1.0
        if self.kernel == "rbf":
            rng = np.random.RandomState(self.seed)
            self.W_ = rng.normal(0.0, np.sqrt(2.0 * self.gamma), (X.shape[1], self.n_components))
            self.b_ = rng.uniform(0.0, 2.0 * np.pi, self.n_components)
        Z = self._map(X)
        n, D = Z.shape
        sw = np.ones(n)
        if self.class_weight == "balanced":
            for c in (0.0, 1.0):
                sw[y == c] = n / (2.0 * (y == c).sum())
        lam = 1.0 / (self.C * n)
        w, b = np.zeros(D), 0.0
        mw, vw, mb, vb = np.zeros(D), np.zeros(D), 0.0, 0.0
        b1, b2, eps = 0.9, 0.999, 1e-8
        w_avg, b_avg, n_avg = np.zeros(D), 0.0, 0
        for t in range(1, self.n_iter + 1):
            viol = ys * (Z @ w + b) < 1.0                       # margin violators
            coef = sw[viol] * ys[viol]
            gw = lam * w - (Z[viol].T @ coef) / n
            gb = -coef.sum() / n
            lr = self.lr / np.sqrt(t)
            mw, vw = b1 * mw + (1 - b1) * gw, b2 * vw + (1 - b2) * gw ** 2
            mb, vb = b1 * mb + (1 - b1) * gb, b2 * vb + (1 - b2) * gb ** 2
            w -= lr * (mw / (1 - b1 ** t)) / (np.sqrt(vw / (1 - b2 ** t)) + eps)
            b -= lr * (mb / (1 - b1 ** t)) / (np.sqrt(vb / (1 - b2 ** t)) + eps)
            if t > self.n_iter // 2:                            # average the 2nd half (less noise)
                w_avg, b_avg, n_avg = w_avg + w, b_avg + b, n_avg + 1
        self.w_, self.bias_ = w_avg / n_avg, b_avg / n_avg
        return self

    def score(self, X):                                         # raw signed margin
        return self._map(X) @ self.w_ + self.bias_

    decision_function = score


# %% tuning
def cross_val_scores(make_model, X, y, n_splits=5, seed=SEED):
    """Out-of-fold raw scores: every training row is scored by a model that never saw it."""
    oof = np.zeros(len(y))
    for tr, va in stratified_kfold(y, n_splits, seed):
        model = make_model()
        model.fit(X[tr], y[tr])
        oof[va] = model.score(X[va])
    return oof


def tune_model(model_cls, grid, X, y, fixed=None, platt=False, n_splits=5, verbose=True):
    """
    Grid search with stratified K-fold on the TRAINING data only.
    For each candidate: out-of-fold probabilities -> best-F1 threshold -> CV F1 / AUC.
    The candidate with the highest CV F1 wins (ties broken by AUC).
    """
    keys = list(grid)
    rows, best = [], None
    for values in product(*[grid[k] for k in keys]):
        params = dict(zip(keys, values))
        params.update(fixed or {})
        oof = cross_val_scores(lambda: model_cls(**params), X, y, n_splits)
        ab = fit_platt(oof, y) if platt else None
        prob = platt_apply(oof, *ab) if platt else oof
        thr, f1 = best_f1_threshold(y, prob)
        row = dict(params)
        row.update({"cv_f1": f1, "cv_auc": roc_auc_score(y, prob),
                    "cv_acc": accuracy_score(y, (prob >= thr).astype(int)), "threshold": thr})
        rows.append(row)
        if best is None or (f1, row["cv_auc"]) > (best["cv_f1"], best["cv_auc"]):
            best = dict(row, _params=params, _platt=ab)
        if verbose:
            print("   ", {k: params[k] for k in keys}, "-> CV F1=%.4f  AUC=%.4f" % (f1, row["cv_auc"]))
    table = pd.DataFrame(rows).sort_values(["cv_f1", "cv_auc"], ascending=False).reset_index(drop=True)
    return {"best_params": best["_params"], "threshold": best["threshold"],
            "platt": best["_platt"], "table": table}


def fit_final(model_cls, tuned, X, y):
    """Refit the winning configuration on the full training set and attach its threshold."""
    model = model_cls(**tuned["best_params"])
    model.fit(X, y)
    model.threshold_ = tuned["threshold"]
    if tuned["platt"] is not None:
        model.platt_ = tuned["platt"]
    return model
