import numpy as np


def load_xy_from_txt(path: str):
    """
    Loads X, y from a text/csv file where last column is label.
    Works for:
      - comma-separated CSV with header (like IRIS.csv, iris_test.csv)
      - space-separated TXT without header
      - labels as ints OR strings (e.g., Iris-setosa)
    """
    # Detect delimiter from first line
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    delim = "," if "," in first_line else None  # None => whitespace
    # Detect header (if first token is not a number)
    first_token = first_line.split(",")[0].split()[0]
    has_header = True
    try:
        float(first_token)
        has_header = False
    except Exception:
        has_header = True

    data = np.genfromtxt(
        path,
        delimiter=delim,
        dtype=str,
        skip_header=1 if has_header else 0
    )

    # If file has only 1 row, genfromtxt may return 1D
    if data.ndim == 1:
        data = data[None, :]

    X = data[:, :-1].astype(float)
    y_raw = data[:, -1]

    # If labels are numeric, convert to int
    try:
        y = y_raw.astype(float)
        if np.allclose(y, np.round(y)):
            y = np.round(y).astype(int)
        return X, y
    except Exception:
        pass

    # Otherwise labels are strings -> map to ints
    classes = sorted(np.unique(y_raw))
    mapping = {c: i for i, c in enumerate(classes)}
    y = np.array([mapping[v] for v in y_raw], dtype=int)
    return X, y


class KNNClassifier:
    def __init__(self, k=3):
        self.k = int(k)

    def fit(self, X_train, y_train):
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train)
        return self

    def _distance_matrix(self, X):
        X = np.asarray(X, dtype=float)           # (m, d)
        Y = self.X_train                         # (n, d)

        X2 = np.sum(X * X, axis=1)[None, :]      # (1, m)
        Y2 = np.sum(Y * Y, axis=1)[:, None]      # (n, 1)
        XY = Y @ X.T                             # (n, m)

        dist2 = Y2 - 2.0 * XY + X2               # (n, m)
        dist2 = np.maximum(dist2, 0.0)
        return np.sqrt(dist2)

    def predict(self, X):
        dist = self._distance_matrix(X)          # (n_train, m_test)
        nn_idx = np.argsort(dist, axis=0)[: self.k, :]   # (k, m)
        nn_labels = self.y_train[nn_idx]         # (k, m)

        # majority vote per column
        y_pred = []
        for j in range(nn_labels.shape[1]):
            vals, counts = np.unique(nn_labels[:, j], return_counts=True)
            y_pred.append(vals[np.argmax(counts)])
        return np.array(y_pred)
