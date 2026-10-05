from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

DISPLAY = {"stateful_puzzle": "StatefulPuzzle", "graph_nav": "GraphNav", "tool_dag": "ToolDAG"}
BACKDROP = ("horizon", "branching", "observation", "mutation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np

    report = json.loads(args.report.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 400})

    def save(fig, stem):
        fig.savefig(args.output / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(args.output / f"{stem}.png", bbox_inches="tight", dpi=400)
        plt.close(fig)

    def slug(values):
        return re.sub(r"[^A-Za-z0-9_-]+", "_", "_".join(map(str, values)))

    facets = defaultdict(list)
    for cell in report["cells"]:
        facets[(cell["env_name"], *(cell[k] for k in BACKDROP))].append(cell)
    for facet, cells in sorted(facets.items()):
        models = sorted({c["model"] for c in cells})
        fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models) + .3, 4.0),
                                 squeeze=False, layout="constrained")
        for ax, model in zip(axes[0], models):
            rows = [c for c in cells if c["model"] == model]
            sizes = sorted({c["state_size"] for c in rows})
            deps = sorted({c["state_dependency"] for c in rows})
            values = np.full((len(sizes), len(deps)), np.nan)
            for row in rows:
                i, j = sizes.index(row["state_size"]), deps.index(row["state_dependency"])
                if row["success_rate"] is not None:
                    values[i, j] = row["success_rate"]
                    mark = "" if row["complete"] else "*"
                    ax.text(j, i, f"{row['success_rate']:.2f}{mark}", ha="center", va="center", fontsize=10)
            cmap = plt.get_cmap("RdYlGn").copy()
            cmap.set_bad("#E6E6E6")
            im = ax.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect="auto")
            for test in report["cliff_tests"]:
                if not test.get("significant_bonferroni") or test["env_name"] != facet[0] or test["model"] != model:
                    continue
                if tuple(test[k] for k in BACKDROP) != facet[1:]:
                    continue
                if test["axis"] == "state_size":
                    y = sizes.index(test["lower"]) + .5
                    x = deps.index(test["state_dependency"])
                    ax.plot([x - .47, x + .47], [y, y], color="#202020", lw=2.4)
                else:
                    x = deps.index(test["lower"]) + .5
                    y = sizes.index(test["state_size"])
                    ax.plot([x, x], [y - .47, y + .47], color="#202020", lw=2.4)
            finite = [r for r in rows if r["complete"] and r["success_rate"] is not None]

            if finite and min(r["success_rate"] for r in finite) <= .5 <= max(r["success_rate"] for r in finite):
                c = min(finite, key=lambda r: (abs(r["success_rate"] - .5), r["state_size"], r["state_dependency"]))
                ax.add_patch(Rectangle((deps.index(c["state_dependency"]) - .45,
                                        sizes.index(c["state_size"]) - .45), .9, .9,
                                       fill=False, edgecolor="#7B3294", linewidth=2, linestyle="--"))
            ax.set(xticks=range(len(deps)), xticklabels=deps, yticks=range(len(sizes)), yticklabels=sizes,
                   xlabel="State dependency (SD)", ylabel="State size (SS)", title=model)
        fig.colorbar(im, ax=list(axes[0]), label="Success rate", shrink=.85)
        backdrop = f"T={facet[1]}, branching={facet[2]}, {facet[3]}, {facet[4]}"
        fig.suptitle(f"{DISPLAY.get(facet[0], facet[0])} | {backdrop}")
        fig.supxlabel("Black: corrected cliff; purple: cell nearest 0.5; *: incomplete cell", fontsize=8)
        save(fig, "phase_" + slug(facet))


    temporal = defaultdict(list)
    for row in report["failure_order"]:
        temporal[tuple(row[k] for k in BACKDROP)].append(row)
    for backdrop, rows in sorted(temporal.items()):
        rows = [row for row in rows if row["paired_n"]]
        if not rows:
            continue
        labels = [f"{DISPLAY.get(r['env_name'], r['env_name'])}\n{r['model']} (n={r['paired_n']})" for r in rows]
        fig, ax = plt.subplots(figsize=(8.5, max(3, .65 * len(rows))), layout="constrained")
        left = np.zeros(len(rows))
        for name, title, color in (("world_first_pct", "World first", "#377EB8"),
                                    ("same_step_pct", "Same step", "#BDBDBD"),
                                    ("action_first_pct", "Action first", "#E66101")):
            values = np.array([r[name] for r in rows])
            ax.barh(range(len(rows)), values, left=left, label=title, color=color, height=.7)
            for i, (v, start) in enumerate(zip(values, left)):
                if v >= 6:
                    ax.text(start + v / 2, i, f"{v:.1f}%", ha="center", va="center", fontsize=9)
            left += values
        ax.set(yticks=range(len(rows)), yticklabels=labels, xlim=(0, 100), xlabel="Paired collapsed rollouts (%)")
        ax.invert_yaxis()
        ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.12), ncol=3, frameon=False)
        save(fig, "failure_order_" + slug(backdrop))

    trajectories = defaultdict(list)
    for row in report["fidelity_before_action"]:
        trajectories[tuple(row[k] for k in BACKDROP)].append(row)
    for backdrop, rows in sorted(trajectories.items()):
        environments = sorted({r["env_name"] for r in rows})
        fig, axes = plt.subplots(1, len(environments), figsize=(4 * len(environments), 3.4),
                                 squeeze=False, layout="constrained")
        for ax, env in zip(axes[0], environments):
            for model in sorted({r["model"] for r in rows if r["env_name"] == env}):
                line = sorted([r for r in rows if r["env_name"] == env and r["model"] == model], key=lambda r: r["relative_step"])
                ax.plot([r["relative_step"] for r in line],
                        [r["mean_fidelity"] if r["mean_fidelity"] is not None else np.nan for r in line],
                        marker="o", linewidth=2, label=model)
            ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
            ax.set(title=DISPLAY.get(env, env), ylim=(0, 1.02), xticks=[-3, -2, -1, 0],
                   xlabel="Steps relative to first invalid action", ylabel="World-state fidelity")
            ax.legend(frameon=False, fontsize=8)
            ax.grid(axis="y", alpha=.2)
        save(fig, "fidelity_before_action_" + slug(backdrop))

    for estimate in report["critical_points"]:
        rows = sorted([c for c in report["cells"] if all(c[k] == estimate[k] for k in ("env_name", "model", "state_dependency", *BACKDROP))], key=lambda c: c["state_size"])
        x, y = [r["state_size"] for r in rows], [r["success_rate"] for r in rows]
        fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
        ax.plot(x, y, color="#BD3037", marker="o", linewidth=2)
        ax.fill_between(x, [r["ci_low"] for r in rows], [r["ci_high"] for r in rows], color="#BD3037", alpha=.16, label="95% Wilson interval")
        ax.axhline(.5, color="#777777", linestyle="--", linewidth=1)
        if estimate["state_size_star"] is not None:
            ax.axvline(estimate["state_size_star"], color="#BD3037", linestyle="--", label=f"SS*={estimate['state_size_star']:.2f}")
        ax.set(xlabel="State size (SS)", ylabel="Success rate", ylim=(-.02, 1.02),
               title=f"{DISPLAY[estimate['env_name']]} | {estimate['model']} | SD=1")
        ax.legend(frameon=False, fontsize=8)
        save(fig, "critical_" + slug([estimate[k] for k in ("env_name", "model", *BACKDROP)]))


if __name__ == "__main__":
    main()
