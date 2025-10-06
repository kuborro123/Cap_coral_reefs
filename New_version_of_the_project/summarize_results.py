import pandas as pd
import re

CSV = "metrics_summary.csv"   # change if you used a different filename
RUN_COL = "run"               # column with run names (e.g., baseline_s0)

# Try to be robust to column naming
df = pd.read_csv(CSV)

# Heuristics for column names in your CSV (adjust if needed)
def pick(col_regex):
    cols = [c for c in df.columns if re.search(col_regex, c, re.I)]
    if not cols:
        raise SystemExit(f"Couldn't find a column matching /{col_regex}/ in {df.columns.tolist()}")
    return cols[0]

soft_col = pick(r"(mask)?.*mAP50.*95.*soft")   # e.g., 'Mask(mAP50-95)_soft_coral'
hard_col = pick(r"(mask)?.*mAP50.*95.*hard")   # e.g., 'Mask(mAP50-95)_hard_coral'
run_col  = RUN_COL

# Add macro if missing
macro_col = None
for c in df.columns:
    if re.search(r"(mask)?.*mAP50.*95.*macro", c, re.I):
        macro_col = c; break
if macro_col is None:
    macro_col = "mask_mAP50-95_macro"
    df[macro_col] = (df[soft_col] + df[hard_col]) / 2

# Pick a baseline row (first name containing 'baseline')
base_idx = df[run_col].str.contains("baseline", case=False, na=False)
if not base_idx.any():
    raise SystemExit("No 'baseline' row found in run names. Make sure one run is named like 'baseline_s0'.")
base = df.loc[base_idx].iloc[0]

# Build a tidy view
view = df[[run_col, soft_col, hard_col, macro_col]].copy()
view["Δ_soft"]  = view[soft_col]  - base[soft_col]
view["Δ_hard"]  = view[hard_col]  - base[hard_col]
view["Δ_macro"] = view[macro_col] - base[macro_col]
view = view.sort_values(macro_col, ascending=False)

print("\n=== Ranked by Macro mAP50–95 (descending) ===")
print(view.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nBaseline reference:")
print(base[[soft_col, hard_col, macro_col]].to_string())

