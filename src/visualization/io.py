# @author: Inigo Martinez Jimenez
# Figure I/O helpers: save figures and animations to disk.

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(
    fig,
    output_dir: Path | str,
    stem: str,
    *,
    formats: tuple[str, ...] = ("png",),
    dpi: int | None = None,
    facecolor: str | None = None,
    close: bool = True,
) -> list[Path]:
    output_dir = ensure_dir(output_dir)
    paths: list[Path] = []
    for fmt in formats:
        out_path = output_dir / f"{stem}.{fmt}"
        fig.savefig(
            out_path,
            dpi=dpi or plt.rcParams["savefig.dpi"],
            facecolor=facecolor or plt.rcParams["savefig.facecolor"],
            edgecolor=plt.rcParams["savefig.edgecolor"],
            bbox_inches="tight",
            pad_inches=0.25,
        )
        paths.append(out_path)
    if close:
        plt.close(fig)
    return paths


def save_animation(
    anim,
    output_dir: Path | str,
    stem: str,
    *,
    fps: int = 18,
    dpi: int = 150,
) -> Path:
    output_dir = ensure_dir(output_dir)
    out_path = output_dir / f"{stem}.gif"
    anim.save(out_path, writer="pillow", fps=fps, dpi=dpi,
              savefig_kwargs={"facecolor": plt.rcParams["savefig.facecolor"]})
    return out_path
