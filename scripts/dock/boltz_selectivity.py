"""
Pair the two Boltz-2 runs and rank by predicted TgCPL-over-HsCPL selectivity.
Expects a tgcpl/ and an hscpl/ subdirectory under the output root; nesting
below that does not matter, since affinity JSONs are found by glob and the
ligand id comes from the filename, which is the join key.

affinity_pred_value is log10(IC50 / uM), so LOWER is stronger and
d_logIC50 = hscpl - tgcpl, positive meaning TgCPL-preferring.

The controls print first on purpose. Boltz-2's affinity head was trained to
rank actives against a single target, so differencing two independent
predictions across two homologues is off-label and the two heads need not
share a zero. Four leads with measured IC50s cannot validate that, but they
can invalidate it: a flat or inverted control correlation means the rankings
below are noise.

Nothing here models the covalent bond to Cys31/Cys25 -- these are non-covalent
poses of a covalent inhibitor.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tgcpl lives under scripts/

from tgcpl.leads import LEADS  # noqa: E402

TARGETS = ("tgcpl", "hscpl")


def find_affinities(target_root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(target_root.rglob("affinity_*.json")):
        lid = path.stem[len("affinity_"):]
        try:
            out[lid] = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"  skipping {path.name}: {e}", file=sys.stderr)
    return out


def find_confidence(target_root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(target_root.rglob("confidence_*_model_0.json")):
        lid = path.stem[len("confidence_"):-len("_model_0")]
        try:
            out[lid] = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
    return out


def load_ligand_table(path: Path | None) -> dict[str, dict]:
    if path is None or not path.is_file():
        return {}
    rows: dict[str, dict] = {}
    header = None
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if header is None:
            header = fields
            continue
        rec = dict(zip(header, fields))
        rows[rec["ligand_id"]] = rec
    return rows


def experimental_d(lid: str) -> float | None:
    """Measured log10 selectivity for a control, from the leads table."""
    if not lid.startswith("CTRL_"):
        return None
    lead = LEADS.get(lid[len("CTRL_"):])
    if not lead:
        return None
    tg, hs = lead.get("tgcpl_ic50_nM"), lead.get("hscpl_ic50_nM")
    if not tg or not hs:
        return None
    return math.log10(hs / tg)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation, without requiring scipy. None if degenerate."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:  # average ranks within ties
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


COLUMNS = [
    "rank", "ligand_id", "d_logIC50", "fold_pred", "tgcpl_logIC50",
    "hscpl_logIC50", "tgcpl_pbind", "hscpl_pbind", "tgcpl_iptm", "hscpl_iptm",
    "is_control", "exp_d_logIC50", "smiles",
]


def fmt(v) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("out_root", type=Path,
                    help="Directory holding the tgcpl/ and hscpl/ Boltz outputs")
    ap.add_argument("--ligands", type=Path, default=None,
                    help="boltz_in/ligands.tsv, to carry SMILES through")
    ap.add_argument("--min-iptm", type=float, default=0.0,
                    help="Drop ligands whose pose confidence is below this on "
                         "either target. A difference of two bad poses is not "
                         "a selectivity estimate")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", type=Path, default=None,
                    help="Default: <out_root>/selectivity_docking.tsv")
    args = ap.parse_args()

    aff, conf = {}, {}
    for t in TARGETS:
        root = args.out_root / t
        if not root.is_dir():
            raise SystemExit(
                f"missing {root}\n"
                f"Expected {args.out_root}/tgcpl and {args.out_root}/hscpl."
            )
        aff[t] = find_affinities(root)
        conf[t] = find_confidence(root)
        print(f"{t:<6} {len(aff[t])} affinity predictions under {root}")

    ligands = load_ligand_table(args.ligands)

    both = sorted(set(aff["tgcpl"]) & set(aff["hscpl"]))
    only_tg = sorted(set(aff["tgcpl"]) - set(aff["hscpl"]))
    only_hs = sorted(set(aff["hscpl"]) - set(aff["tgcpl"]))
    print(f"\nPaired : {len(both)}")
    if only_tg or only_hs:
        print(f"  unpaired: {len(only_tg)} TgCPL-only, {len(only_hs)} HsCPL-only"
              f" -- those targets' runs are out of step, check for failures")
    if not both:
        raise SystemExit("nothing to compare")

    rows = []
    for lid in both:
        tg = aff["tgcpl"][lid].get("affinity_pred_value")
        hs = aff["hscpl"][lid].get("affinity_pred_value")
        if tg is None or hs is None:
            continue
        tg_c = conf["tgcpl"].get(lid, {})
        hs_c = conf["hscpl"].get(lid, {})
        meta = ligands.get(lid, {})
        d = hs - tg
        rows.append({
            "ligand_id": lid,
            "d_logIC50": d,
            # 10**d overflows the column for a wild prediction; cap the display.
            "fold_pred": min(10 ** d, 1e6) if d > 0 else 10 ** d,
            "tgcpl_logIC50": tg,
            "hscpl_logIC50": hs,
            "tgcpl_pbind": aff["tgcpl"][lid].get("affinity_probability_binary"),
            "hscpl_pbind": aff["hscpl"][lid].get("affinity_probability_binary"),
            "tgcpl_iptm": tg_c.get("ligand_iptm", tg_c.get("iptm")),
            "hscpl_iptm": hs_c.get("ligand_iptm", hs_c.get("iptm")),
            "is_control": 1 if lid.startswith("CTRL_") else 0,
            "exp_d_logIC50": experimental_d(lid),
            "smiles": meta.get("smiles", "NA"),
        })

    if args.min_iptm > 0:
        before = len(rows)
        rows = [
            r for r in rows
            if (r["tgcpl_iptm"] is None or r["tgcpl_iptm"] >= args.min_iptm)
            and (r["hscpl_iptm"] is None or r["hscpl_iptm"] >= args.min_iptm)
        ]
        print(f"Pose confidence >= {args.min_iptm}: {len(rows)}/{before}")

    controls = [r for r in rows if r["is_control"] and r["exp_d_logIC50"] is not None]
    print("\n" + "=" * 68)
    print("CONTROLS — do the known leads come out in the measured order?")
    print("=" * 68)
    if not controls:
        print("  No controls in this run. Re-run boltz_prep.py without")
        print("  --no-controls before trusting anything below.")
    else:
        print(f"  {'lead':<14}{'pred d':>9}{'pred fold':>11}"
              f"{'exp d':>9}{'exp fold':>11}")
        print(f"  {'-' * 54}")
        for r in sorted(controls, key=lambda r: -r["exp_d_logIC50"]):
            print(f"  {r['ligand_id']:<14}{r['d_logIC50']:>9.2f}"
                  f"{10 ** r['d_logIC50']:>11.1f}"
                  f"{r['exp_d_logIC50']:>9.2f}"
                  f"{10 ** r['exp_d_logIC50']:>11.1f}")
        rho = spearman([r["d_logIC50"] for r in controls],
                       [r["exp_d_logIC50"] for r in controls])
        print()
        if rho is None:
            print("  Too few controls to correlate.")
        else:
            print(f"  Spearman rho = {rho:+.2f} over {len(controls)} leads.")
            if rho <= 0:
                print("  FLAT OR INVERTED. The predicted selectivity does not")
                print("  track the measured selectivity on compounds where the")
                print("  answer is known. Do not rank candidates on this.")
            else:
                print("  Positive, but n=4 congeneric compounds. This is a")
                print("  sanity check that the pipeline is wired up, not")
                print("  evidence that it generalises to new scaffolds.")

    rows.sort(key=lambda r: -r["d_logIC50"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    out = args.out or args.out_root / "selectivity_docking.tsv"
    with out.open("w") as fh:
        fh.write("# boltz_selectivity.py\n")
        fh.write("# d_logIC50 = hscpl - tgcpl (log10 IC50/uM); "
                 "positive = TgCPL-preferring\n")
        fh.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            fh.write("\t".join(fmt(r.get(c)) for c in COLUMNS) + "\n")
    print(f"\nWrote {len(rows)} rows -> {out}")

    gen = [r for r in rows if not r["is_control"]][: args.top]
    print(f"\nTop {len(gen)} generated molecules by predicted selectivity:")
    print(f"  {'#':<5}{'id':<10}{'d':>7}{'fold':>9}{'Tg':>7}{'Hs':>7}"
          f"{'iptmTg':>8}  SMILES")
    print(f"  {'-' * 86}")
    for r in gen:
        smi = r["smiles"]
        smi = smi if len(smi) <= 38 else smi[:35] + "..."
        print(f"  {r['rank']:<5}{r['ligand_id']:<10}{r['d_logIC50']:>7.2f}"
              f"{10 ** r['d_logIC50']:>9.1f}{r['tgcpl_logIC50']:>7.2f}"
              f"{r['hscpl_logIC50']:>7.2f}{fmt(r['tgcpl_iptm']):>8}  {smi}")


if __name__ == "__main__":
    main()
