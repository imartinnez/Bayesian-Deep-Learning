from __future__ import annotations

import numpy as np
from scipy.stats import norm


STATE_NAMES = ("NORMAL", "ALERTA", "ESTRES")


def _to_1d_float_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _to_state_codes(states: np.ndarray) -> np.ndarray:
    labels = np.asarray(states, dtype=object).reshape(-1)
    mapping = {"NORMAL": 0, "ALERTA": 1, "ESTRES": 2}
    try:
        return np.array([mapping[str(label)] for label in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unknown semaphore state: {exc}") from exc


def _codes_to_states(codes: np.ndarray) -> np.ndarray:
    values = np.asarray(codes, dtype=np.int64).reshape(-1)
    if np.any((values < 0) | (values > 2)):
        raise ValueError("Semaphore codes must be in {0, 1, 2}.")
    return np.array([STATE_NAMES[int(code)] for code in values], dtype=object)


def _binary_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    truth = np.asarray(y_true, dtype=bool).reshape(-1)
    pred = np.asarray(y_pred, dtype=bool).reshape(-1)
    if truth.shape != pred.shape:
        raise ValueError("Binary metric inputs must share the same shape.")

    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _multiclass_macro_f1(y_true_codes: np.ndarray, y_pred_codes: np.ndarray) -> float:
    truth = np.asarray(y_true_codes, dtype=np.int64).reshape(-1)
    pred = np.asarray(y_pred_codes, dtype=np.int64).reshape(-1)
    if truth.shape != pred.shape:
        raise ValueError("Macro-F1 inputs must share the same shape.")

    f1_scores = []
    for code in (0, 1, 2):
        metrics = _binary_precision_recall_f1(truth == code, pred == code)
        f1_scores.append(metrics["f1"])
    return float(np.mean(f1_scores))


def _confusion_matrix(y_true_codes: np.ndarray, y_pred_codes: np.ndarray) -> np.ndarray:
    truth = np.asarray(y_true_codes, dtype=np.int64).reshape(-1)
    pred = np.asarray(y_pred_codes, dtype=np.int64).reshape(-1)
    if truth.shape != pred.shape:
        raise ValueError("Confusion-matrix inputs must share the same shape.")

    matrix = np.zeros((3, 3), dtype=np.int64)
    for actual_code in (0, 1, 2):
        for predicted_code in (0, 1, 2):
            matrix[actual_code, predicted_code] = int(
                np.sum((truth == actual_code) & (pred == predicted_code))
            )
    return matrix


def _apply_stress_hysteresis(
    raw_estres_mask: np.ndarray,
    prob_estres: np.ndarray,
    prob_estres_thr: float,
    stress_persistence: int,
    stress_high_prob_scale: float,
) -> tuple[np.ndarray, float]:
    raw = np.asarray(raw_estres_mask, dtype=bool).reshape(-1)
    probs = np.asarray(prob_estres, dtype=float).reshape(-1)
    if raw.shape != probs.shape:
        raise ValueError("raw_estres_mask and prob_estres must share the same shape.")

    persistence = max(1, int(stress_persistence))
    high_prob_thr = float(min(0.99, prob_estres_thr * stress_high_prob_scale))

    if persistence == 1:
        return raw.copy(), high_prob_thr

    final_mask = np.zeros_like(raw, dtype=bool)
    run_start = None

    for i, is_raw_stress in enumerate(raw):
        if is_raw_stress:
            if run_start is None:
                run_start = i

            run_length = i - run_start + 1
            if run_length >= persistence:
                final_mask[run_start:i + 1] = True
            elif probs[i] >= high_prob_thr:
                final_mask[i] = True
        else:
            run_start = None

    return final_mask, high_prob_thr


def compute_spr_thresholds(
    train_rv: np.ndarray,
    sigma_epist_train: np.ndarray,
    alerta_pct: float = 75.0,
    estres_pct: float = 90.0,
    epist_pct: float = 75.0,
) -> dict:
    train_rv = _to_1d_float_array(train_rv, "train_rv")
    sigma_epist_train = _to_1d_float_array(sigma_epist_train, "sigma_epist_train")

    return {
        "umbral_alerta": float(np.percentile(train_rv, alerta_pct)),
        "umbral_estres": float(np.percentile(train_rv, estres_pct)),
        "umbral_epist": float(np.percentile(sigma_epist_train, epist_pct)),
        "alerta_pct": float(alerta_pct),
        "estres_pct": float(estres_pct),
        "epist_pct": float(epist_pct),
    }


def compute_rolling_rv_thresholds(
    rv_full: np.ndarray,
    n_target: int,
    window: int = 252,
    alerta_pct: float = 75.0,
    estres_pct: float = 90.0,
    min_obs: int = 63,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute time-varying RV thresholds for the last n_target observations.
    Each day t uses only rv_full[max(0, t-window):t] — purely causal, no lookahead.
    """
    rv = np.asarray(rv_full, dtype=float)
    n = len(rv)
    offset = n - n_target
    umbral_alerta = np.empty(n_target)
    umbral_estres = np.empty(n_target)

    for i in range(n_target):
        t = offset + i
        start = max(0, t - window)
        hist = rv[start:t]
        if len(hist) < min_obs:
            hist = rv[max(0, t - min_obs):t]
        if len(hist) == 0:
            hist = rv[:1]
        umbral_alerta[i] = np.percentile(hist, alerta_pct)
        umbral_estres[i] = np.percentile(hist, estres_pct)

    return umbral_alerta, umbral_estres


def realized_risk_states(
    rv_series: np.ndarray,
    umbral_alerta: float,
    umbral_estres: float,
) -> dict:
    rv = _to_1d_float_array(rv_series, "rv_series")
    codes = np.zeros_like(rv, dtype=np.int64)
    codes[rv > np.asarray(umbral_alerta, dtype=float)] = 1
    codes[rv > np.asarray(umbral_estres, dtype=float)] = 2
    return {
        "states": _codes_to_states(codes),
        "state_codes": codes,
    }


def compute_exceedance_probabilities(
    mu_pred: np.ndarray,
    sigma_total: np.ndarray,
    umbral_alerta: float,
    umbral_estres: float,
) -> dict:
    mu = _to_1d_float_array(mu_pred, "mu_pred")
    sigma = np.clip(_to_1d_float_array(sigma_total, "sigma_total"), 1e-8, None)

    if mu.shape != sigma.shape:
        raise ValueError("mu_pred and sigma_total must have the same shape.")

    log_alerta = np.log(np.maximum(np.asarray(umbral_alerta, dtype=float), 1e-12))
    log_estres = np.log(np.maximum(np.asarray(umbral_estres, dtype=float), 1e-12))

    z_alerta = (log_alerta - mu) / sigma
    z_estres = (log_estres - mu) / sigma

    q95_log = mu + 1.645 * sigma
    q95_rv = np.exp(q95_log)

    return {
        "prob_alerta": norm.sf(z_alerta),
        "prob_estres": norm.sf(z_estres),
        "q95_rv": q95_rv,
    }


def compute_spr(
    mu_pred: np.ndarray,
    sigma_total: np.ndarray,
    sigma_epist: np.ndarray,
    umbral_alerta: float,
    umbral_estres: float,
    umbral_epist: float,
    prob_alerta_thr: float = 0.30,
    prob_estres_thr: float = 0.10,
    epist_scale: float = 1.0,
    stress_persistence: int = 1,
    stress_high_prob_scale: float = 2.0,
) -> dict:
    mu = _to_1d_float_array(mu_pred, "mu_pred")
    sigma = _to_1d_float_array(sigma_total, "sigma_total")
    epist = _to_1d_float_array(sigma_epist, "sigma_epist")

    shapes = {mu.shape, sigma.shape, epist.shape}
    if len(shapes) != 1:
        raise ValueError(
            "mu_pred, sigma_total, and sigma_epist must have the same shape. "
            f"Received: {mu.shape}, {sigma.shape}, {epist.shape}"
        )

    exceedance = compute_exceedance_probabilities(
        mu_pred=mu,
        sigma_total=sigma,
        umbral_alerta=umbral_alerta,
        umbral_estres=umbral_estres,
    )
    epist_threshold = float(umbral_epist * epist_scale)

    alerta_mask = (exceedance["prob_alerta"] >= prob_alerta_thr) | (epist >= epist_threshold)
    raw_estres_mask = (exceedance["prob_estres"] >= prob_estres_thr) | (
        (exceedance["prob_alerta"] >= prob_alerta_thr) & (epist >= epist_threshold)
    )
    estres_mask, stress_high_prob_thr = _apply_stress_hysteresis(
        raw_estres_mask=raw_estres_mask,
        prob_estres=exceedance["prob_estres"],
        prob_estres_thr=prob_estres_thr,
        stress_persistence=stress_persistence,
        stress_high_prob_scale=stress_high_prob_scale,
    )

    state_codes = np.zeros_like(mu, dtype=np.int64)
    state_codes[alerta_mask] = 1
    state_codes[estres_mask] = 2

    return {
        "states": _codes_to_states(state_codes),
        "state_codes": state_codes,
        "prob_alerta": exceedance["prob_alerta"],
        "prob_estres": exceedance["prob_estres"],
        "q95_rv": exceedance["q95_rv"],
        "epist_threshold": epist_threshold,
        "raw_estres_mask": raw_estres_mask,
        "stress_high_prob_thr": stress_high_prob_thr,
    }


def evaluate_spr_states(
    actual_rv: np.ndarray,
    predicted_states: np.ndarray,
    umbral_alerta: float,
    umbral_estres: float,
) -> dict:
    rv = _to_1d_float_array(actual_rv, "actual_rv")
    pred_codes = _to_state_codes(predicted_states)
    if rv.shape != pred_codes.shape:
        raise ValueError("actual_rv and predicted_states must share the same shape.")

    actual = realized_risk_states(rv, umbral_alerta=umbral_alerta, umbral_estres=umbral_estres)
    actual_codes = actual["state_codes"]

    alert_metrics = _binary_precision_recall_f1(actual_codes >= 1, pred_codes >= 1)
    stress_metrics = _binary_precision_recall_f1(actual_codes == 2, pred_codes == 2)
    macro_f1 = _multiclass_macro_f1(actual_codes, pred_codes)
    confusion = _confusion_matrix(actual_codes, pred_codes)

    signal_rv_stats = {}
    for label in STATE_NAMES:
        mask = np.asarray(predicted_states, dtype=object) == label
        if np.any(mask):
            signal_rv_stats[label] = {
                "n": int(np.sum(mask)),
                "mean_rv": float(np.mean(rv[mask])),
                "median_rv": float(np.median(rv[mask])),
                "mean_logvol": float(np.mean(np.log(np.clip(rv[mask], 1e-12, None)))),
            }

    return {
        "actual_states": actual["states"],
        "actual_state_codes": actual_codes,
        "predicted_state_codes": pred_codes,
        "confusion_matrix": confusion,
        "macro_f1": float(macro_f1),
        "alert_metrics": alert_metrics,
        "stress_metrics": stress_metrics,
        "actual_alert_rate": float(np.mean(actual_codes >= 1)),
        "predicted_alert_rate": float(np.mean(pred_codes >= 1)),
        "actual_stress_rate": float(np.mean(actual_codes == 2)),
        "predicted_stress_rate": float(np.mean(pred_codes == 2)),
        "signal_rv_stats": signal_rv_stats,
    }


def tune_spr_decision_parameters(
    mu_val: np.ndarray,
    sigma_total_val: np.ndarray,
    sigma_epist_val: np.ndarray,
    val_rv: np.ndarray,
    train_rv: np.ndarray,
    sigma_epist_train: np.ndarray,
    prob_alerta_grid: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50),
    prob_estres_grid: tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    epist_scale_grid: tuple[float, ...] = (0.75, 1.00, 1.25, 1.50, 1.75, 2.00),
    stress_persistence_grid: tuple[int, ...] = (1, 2),
    stress_high_prob_scale_grid: tuple[float, ...] = (1.75, 2.00, 2.50),
    umbral_alerta_val: np.ndarray | None = None,
    umbral_estres_val: np.ndarray | None = None,
) -> dict:
    thresholds = compute_spr_thresholds(
        train_rv=train_rv,
        sigma_epist_train=sigma_epist_train,
    )

    ua_val = umbral_alerta_val if umbral_alerta_val is not None else thresholds["umbral_alerta"]
    ue_val = umbral_estres_val if umbral_estres_val is not None else thresholds["umbral_estres"]

    best: dict | None = None
    for prob_alerta_thr in prob_alerta_grid:
        for prob_estres_thr in prob_estres_grid:
            if prob_estres_thr > prob_alerta_thr:
                continue

            for epist_scale in epist_scale_grid:
                for stress_persistence in stress_persistence_grid:
                    for stress_high_prob_scale in stress_high_prob_scale_grid:
                        spr = compute_spr(
                            mu_pred=mu_val,
                            sigma_total=sigma_total_val,
                            sigma_epist=sigma_epist_val,
                            umbral_alerta=ua_val,
                            umbral_estres=ue_val,
                            umbral_epist=thresholds["umbral_epist"],
                            prob_alerta_thr=prob_alerta_thr,
                            prob_estres_thr=prob_estres_thr,
                            epist_scale=epist_scale,
                            stress_persistence=stress_persistence,
                            stress_high_prob_scale=stress_high_prob_scale,
                        )
                        metrics = evaluate_spr_states(
                            actual_rv=val_rv,
                            predicted_states=spr["states"],
                            umbral_alerta=ua_val,
                            umbral_estres=ue_val,
                        )

                        n_unique = len(np.unique(spr["state_codes"]))
                        actual_alert_rate = metrics["actual_alert_rate"]
                        predicted_alert_rate = metrics["predicted_alert_rate"]
                        actual_stress_rate = metrics["actual_stress_rate"]
                        predicted_stress_rate = metrics["predicted_stress_rate"]

                        stress_floor = max(0.5 * actual_stress_rate, 0.02)
                        stress_ceiling = min(max(2.0 * actual_stress_rate, stress_floor + 1e-8), 0.35)
                        alert_floor = max(0.6 * actual_alert_rate, 0.10)
                        alert_ceiling = min(max(1.4 * actual_alert_rate, alert_floor + 1e-8), 0.85)

                        rate_penalty = (
                            0.30 * max(0.0, stress_floor - predicted_stress_rate)
                            + 0.45 * max(0.0, predicted_stress_rate - stress_ceiling)
                            + 0.10 * max(0.0, alert_floor - predicted_alert_rate)
                            + 0.05 * max(0.0, predicted_alert_rate - alert_ceiling)
                        )
                        diversity_penalty = 0.10 * max(0, 3 - n_unique)
                        zero_stress_penalty = 0.25 if actual_stress_rate > 0.0 and predicted_stress_rate == 0.0 else 0.0
                        persistence_penalty = 0.03 * max(0, stress_persistence - 1)

                        score = (
                            0.45 * metrics["stress_metrics"]["f1"]
                            + 0.10 * metrics["stress_metrics"]["recall"]
                            + 0.10 * metrics["stress_metrics"]["precision"]
                            + 0.20 * metrics["alert_metrics"]["f1"]
                            + 0.15 * metrics["macro_f1"]
                            - rate_penalty
                            - diversity_penalty
                            - zero_stress_penalty
                            - persistence_penalty
                        )

                        candidate = {
                            "prob_alerta_thr": float(prob_alerta_thr),
                            "prob_estres_thr": float(prob_estres_thr),
                            "epist_scale": float(epist_scale),
                            "stress_persistence": int(stress_persistence),
                            "stress_high_prob_scale": float(stress_high_prob_scale),
                            "stress_high_prob_thr": float(spr["stress_high_prob_thr"]),
                            "umbral_epist_decision": float(spr["epist_threshold"]),
                            "validation_score": float(score),
                            "validation_metrics": metrics,
                        }

                        if best is None or candidate["validation_score"] > best["validation_score"]:
                            best = candidate

    if best is None:
        raise RuntimeError("SPR tuning failed to evaluate any candidate decision parameters.")

    return {
        **thresholds,
        **best,
    }


def build_semaphore_risk(
    mu_pred: np.ndarray,
    sigma_total: np.ndarray,
    sigma_epist: np.ndarray,
    train_rv: np.ndarray,
    sigma_epist_train: np.ndarray,
    mu_val: np.ndarray | None = None,
    sigma_total_val: np.ndarray | None = None,
    sigma_epist_val: np.ndarray | None = None,
    val_rv: np.ndarray | None = None,
    decision_params: dict | None = None,
    rv_history: np.ndarray | None = None,
    rolling_window: int = 252,
) -> tuple[np.ndarray, dict]:
    thresholds = compute_spr_thresholds(
        train_rv=train_rv,
        sigma_epist_train=sigma_epist_train,
    )

    n_test = len(mu_pred)

    if rv_history is not None and val_rv is not None:
        n_val = len(val_rv)
        ua_val, ue_val = compute_rolling_rv_thresholds(
            rv_full=rv_history[:-n_test],
            n_target=n_val,
            window=rolling_window,
            alerta_pct=thresholds["alerta_pct"],
            estres_pct=thresholds["estres_pct"],
        )
        ua_test, ue_test = compute_rolling_rv_thresholds(
            rv_full=rv_history,
            n_target=n_test,
            window=rolling_window,
            alerta_pct=thresholds["alerta_pct"],
            estres_pct=thresholds["estres_pct"],
        )
        thresholds["umbral_alerta"] = float(np.mean(ua_test))
        thresholds["umbral_estres"] = float(np.mean(ue_test))
    else:
        ua_val = ue_val = None
        ua_test = thresholds["umbral_alerta"]
        ue_test = thresholds["umbral_estres"]

    tuning = None
    if decision_params is None:
        if all(v is not None for v in (mu_val, sigma_total_val, sigma_epist_val, val_rv)):
            tuning = tune_spr_decision_parameters(
                mu_val=mu_val,
                sigma_total_val=sigma_total_val,
                sigma_epist_val=sigma_epist_val,
                val_rv=val_rv,
                train_rv=train_rv,
                sigma_epist_train=sigma_epist_train,
                umbral_alerta_val=ua_val,
                umbral_estres_val=ue_val,
            )
            decision_params = {
                "prob_alerta_thr": tuning["prob_alerta_thr"],
                "prob_estres_thr": tuning["prob_estres_thr"],
                "epist_scale": tuning["epist_scale"],
                "stress_persistence": tuning["stress_persistence"],
                "stress_high_prob_scale": tuning["stress_high_prob_scale"],
            }
        else:
            decision_params = {
                "prob_alerta_thr": 0.30,
                "prob_estres_thr": 0.10,
                "epist_scale": 1.0,
                "stress_persistence": 1,
                "stress_high_prob_scale": 2.0,
            }

    spr = compute_spr(
        mu_pred=mu_pred,
        sigma_total=sigma_total,
        sigma_epist=sigma_epist,
        umbral_alerta=ua_test,
        umbral_estres=ue_test,
        umbral_epist=thresholds["umbral_epist"],
        prob_alerta_thr=float(decision_params["prob_alerta_thr"]),
        prob_estres_thr=float(decision_params["prob_estres_thr"]),
        epist_scale=float(decision_params["epist_scale"]),
        stress_persistence=int(decision_params["stress_persistence"]),
        stress_high_prob_scale=float(decision_params["stress_high_prob_scale"]),
    )

    meta = {
        **thresholds,
        "prob_alerta_thr": float(decision_params["prob_alerta_thr"]),
        "prob_estres_thr": float(decision_params["prob_estres_thr"]),
        "epist_scale": float(decision_params["epist_scale"]),
        "stress_persistence": int(decision_params["stress_persistence"]),
        "stress_high_prob_scale": float(decision_params["stress_high_prob_scale"]),
        "stress_high_prob_thr": float(spr["stress_high_prob_thr"]),
        "umbral_epist_decision": float(spr["epist_threshold"]),
        "prob_alerta": spr["prob_alerta"],
        "prob_estres": spr["prob_estres"],
        "q95_rv": spr["q95_rv"],
        "state_codes": spr["state_codes"],
    }
    if rv_history is not None:
        meta["umbral_alerta_series"] = ua_test
        meta["umbral_estres_series"] = ue_test
    if tuning is not None:
        meta["tuning"] = tuning
    return spr["states"], meta
