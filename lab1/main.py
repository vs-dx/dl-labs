"""ЛР1: ручний backpropagation і дві незалежні перевірки градієнтів."""

import argparse
import numpy as np
import torch
from sklearn.datasets import load_iris



def prepare_data():
    iris = load_iris()
    X = np.asarray(iris.data, dtype=np.float64)
    y = np.asarray(iris.target, dtype=np.int64)
    rng = np.random.default_rng(0)
    train_idx, test_idx = [], []
    for c in (0, 1, 2):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        train_idx.extend(idx[:35])
        test_idx.extend(idx[35:])
    X_train, X_test = X[train_idx], X[test_idx]
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, y[train_idx], X_test, y[test_idx]



def init_params():
    rng = np.random.default_rng(0)
    W1 = rng.normal(0, np.sqrt(2 / 4), size=(4, 8))
    W2 = rng.normal(0, np.sqrt(2 / (8 + 3)), size=(8, 3))
    return {"W1": W1, "b1": np.zeros(8), "W2": W2, "b2": np.zeros(3)}



def forward(X, y, p):
    Z1 = X @ p["W1"] + p["b1"]
    A1 = np.maximum(Z1, 0)
    Z2 = A1 @ p["W2"] + p["b2"]  
    z = Z2 - Z2.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    P = np.exp(log_probs)
    loss = -log_probs[np.arange(len(y)), y].mean()
    return loss, (X, Z1, A1, Z2, P)



def backward(cache, y, p, buggy=False):
    X, Z1, A1, Z2, P = cache
    Y = np.eye(3, dtype=np.float64)[y]
    delta2 = P - Y
    if not buggy:
        delta2 = delta2 / len(y)  
    dW2 = A1.T @ delta2
    db2 = delta2.sum(axis=0)
    delta1 = (delta2 @ p["W2"].T) * (Z1 > 0)
    dW1 = X.T @ delta1
    db1 = delta1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}



def torch_reference(X, y, p):
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8, dtype=torch.float64),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3, dtype=torch.float64),
    )
    with torch.no_grad():
        for layer, name in ((model[0], "1"), (model[2], "2")):
            layer.weight.copy_(torch.tensor(p["W" + name].T, dtype=torch.float64))
            layer.bias.copy_(torch.tensor(p["b" + name], dtype=torch.float64))
    logits = model(torch.tensor(X, dtype=torch.float64))
    loss = torch.nn.functional.cross_entropy(
        logits, torch.tensor(y, dtype=torch.int64), reduction="mean"
    )
    loss.backward()
    grads = {
        "W1": model[0].weight.grad.detach().numpy().T.copy(),
        "b1": model[0].bias.grad.detach().numpy().copy(),
        "W2": model[2].weight.grad.detach().numpy().T.copy(),
        "b2": model[2].bias.grad.detach().numpy().copy(),
    }
    return loss.item(), grads



def numerical_checks(X, y, p, manual):
    eps = 1e-6
    _, initial_cache = forward(X, y, p)
    initial_mask = initial_cache[1] > 0
    selected = (("W1", (0, 0)), ("b1", (0,)), ("W2", (0, 0)), ("b2", (0,)))
    rows = []
    for name, index in selected:
        weights = p[name]
        original = weights[index]
        try:
            weights[index] = original + eps
            loss_plus, plus_cache = forward(X, y, p)
            weights[index] = original - eps
            loss_minus, minus_cache = forward(X, y, p)
        finally:
            weights[index] = original  
        numerical = (loss_plus - loss_minus) / (2 * eps)
        manual_value = manual[name][index]
        diff = abs(numerical - manual_value)
        same_mask = (np.array_equal(plus_cache[1] > 0, initial_mask)
                     and np.array_equal(minus_cache[1] > 0, initial_mask))
        passed = bool(np.isfinite(diff) and diff <= 1e-7 and same_mask)
        rows.append((f"{name}[{', '.join(map(str, index))}]", manual_value, numerical, diff, passed))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buggy", action="store_true", help="дослід із пропуском 1/N")
    args = parser.parse_args()

    X, y, X_test, y_test = prepare_data()
    p = init_params()
    initial = {name: value.copy() for name, value in p.items()}
    loss, cache = forward(X, y, p)
    manual = backward(cache, y, p, buggy=args.buggy)
    torch_loss, torch_grads = torch_reference(X, y, p)

    print(f"Режим: {'ПОМИЛКОВИЙ' if args.buggy else 'ПРАВИЛЬНИЙ'}")
    print(f"Train/test: {len(X)}/{len(X_test)}; класи train: {np.bincount(y).tolist()}")
    print("Форми X, Z1, A1, Z2, P:", [a.shape for a in cache])
    print(f"Loss NumPy:   {loss:.16f}")
    print(f"Loss PyTorch: {torch_loss:.16f}")
    print("\nПорівняння з PyTorch (допуск 1e-12):")
    print("Величина | Макс. абсолютна різниця | Результат")
    loss_diff = abs(loss - torch_loss)
    print(f"Loss       | {loss_diff:.16e} | {'PASS' if loss_diff <= 1e-12 else 'FAIL'}")
    torch_passes = []
    for name in ("W1", "b1", "W2", "b2"):
        diff = np.max(np.abs(manual[name] - torch_grads[name]))
        passed = bool(np.isfinite(diff) and diff <= 1e-12)
        torch_passes.append(passed)
        print(f"d{name:2s}        | {diff:.16e} | {'PASS' if passed else 'FAIL'}")

    print("\nЦентральні різниці (epsilon=1e-6, допуск 1e-7):")
    print("Параметр | Backward | Чисельно | Різниця | Результат")
    fd_rows = numerical_checks(X, y, p, manual)
    for name, analytical, numerical, diff, passed in fd_rows:
        print(f"{name} | {analytical:.12e} | {numerical:.12e} | {diff:.12e} | {'PASS' if passed else 'FAIL'}")

    assert len(X) == 105 and len(X_test) == 45
    assert np.array_equal(np.bincount(y), [35, 35, 35])
    assert np.array_equal(np.bincount(y_test), [15, 15, 15])
    assert X.dtype == np.float64 and X_test.dtype == np.float64
    assert all(v.dtype == np.float64 and np.isfinite(v).all() for v in p.values())
    assert all(np.array_equal(p[name], initial[name]) for name in p)
    assert np.isfinite(loss) and np.isfinite(torch_loss) and loss_diff <= 1e-12
    if args.buggy:
        correct = backward(cache, y, p)
        assert all(np.allclose(manual[name], 105 * correct[name], atol=1e-12, rtol=0)
                   for name in p)
        assert not any(torch_passes) and not any(row[-1] for row in fd_rows)
        print("\nРезультат досліду: 8 очікуваних FAIL виявлено. PASS")
    else:
        assert all(torch_passes) and all(row[-1] for row in fd_rows)
        print("\nВисновок: усі 5 звірок PyTorch та 4 чисельні перевірки пройдено.")


if __name__ == "__main__":
    main()
