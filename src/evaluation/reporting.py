SEP = "=" * 80
SUBSEP = "-" * 80


def print_header(title: str) -> None:
    print("\n" + SEP)
    print(title)
    print(SEP)


def print_kv(label: str, value, indent: int = 0) -> None:
    pad = " " * indent
    print(f"{pad}{label:<28} {value}")


def print_metric_block(title: str, metrics: dict, keys: list[str]) -> None:
    print("\n" + title)
    print(SUBSEP)
    for key in keys:
        value = metrics.get(key, float("nan"))
        if isinstance(value, (int, float)):
            print(f"{key:<28} {value:.4f}")
        else:
            print(f"{key:<28} {value}")


def print_regime_block(title: str, regime_results: dict) -> None:
    print("\n" + title)
    print(SUBSEP)
    print(f"{'Regime':<10}{'n':>6}{'cov90':>10}{'crps':>10}{'nll':>10}{'sharp90':>10}")
    for regime in ("LOW", "MED", "HIGH"):
        r = regime_results[regime]
        print(
            f"{regime:<10}"
            f"{r['n_obs']:>6d}"
            f"{r['coverage_90']:>10.4f}"
            f"{r['crps']:>10.4f}"
            f"{r['nll']:>10.4f}"
            f"{r['sharpness_90']:>10.4f}"
        )


def print_artifacts(paths: dict[str, str]) -> None:
    print("\nArtifacts")
    print(SUBSEP)
    for label, path in paths.items():
        print(f"{label:<28} {path}")