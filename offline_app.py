import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd

# ------------------------------------------------------------
# Make local package imports work both in normal Python runs
# and after PyInstaller packaging.
# ------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_DIR = getattr(sys, "_MEIPASS", APP_DIR)

for path in (APP_DIR, BUNDLE_DIR):
    if path and path not in sys.path:
        sys.path.insert(0, path)

from lcc.dataset import build_df
from lcc.models import (
    build_models,
    predict_f3_from_wb,
    predict_f7_from_wb,
    predict_f14_from_wb,
    implied_f28_from_wb,
    predict_wb_from_f28_curve,
)
from lcc.design import design_mix_from_strengths_min


APP_TITLE = "Low-Carbon Concrete Mix Design Tool"
FAMILY_OPTIONS = ["P1", "F2", "F4", "F5", "S3", "S5", "S6", "T1", "T2"]
AGE_MAP = {"3 Day": 3, "7 Day": 7, "14 Day": 14}

DISPLAY_ORDER = [
    ("inputs.min_early_MPa", "Minimum early-age strength (MPa)"),
    ("inputs.early_age_days", "Early-age requirement (days)"),
    ("inputs.min_28d_MPa", "Minimum 28-day strength (MPa)"),
    ("inputs.binder_family", "Concrete family"),
    ("predicted_parameters.water_binder_ratio", "Water–binder ratio"),
    ("predicted_parameters.water_kg_m3", "Water (kg/m³)"),
    ("predicted_parameters.binder_total_kg_m3", "Total binder (kg/m³)"),
    ("binder_exact.Cement", "Cement (kg/m³)"),
    ("binder_exact.GGBFS", "GGBFS (kg/m³)"),
    ("binder_exact.Fly Ash", "Fly Ash (kg/m³)"),
    ("admixture_split_kg_m3.Plastiment 30", "Plastiment 30 (kg/m³)"),
    ("admixture_split_kg_m3.ECO WR", "ECO WR (kg/m³)"),
    ("admixture_split_kg_m3.Retarder", "Retarder (kg/m³)"),
    ("aggregates_exact.20mm Aggregate", "20mm Aggregate (kg/m³)"),
    ("aggregates_exact.10mm Aggregate", "10mm Aggregate (kg/m³)"),
    ("aggregates_exact.Man Sand", "Manufactured Sand (kg/m³)"),
    ("aggregates_exact.Natural Sand", "Natural Sand (kg/m³)"),
    ("predicted_parameters.fresh_density_target_kg_m3", "Fresh density target (kg/m³)"),
    ("predicted_parameters.air_percent", "Air content (%)"),
    ("embodied_carbon.EC_A1", "Embodied carbon A1 (kgCO₂-e/m³)"),
    ("embodied_carbon.EC_A2", "Embodied carbon A2 (kgCO₂-e/m³)"),
    ("embodied_carbon.EC_A3", "Embodied carbon A3 (kgCO₂-e/m³)"),
    ("embodied_carbon.EC_total", "Embodied carbon total (kgCO₂-e/m³)"),
    ("totals.sum_all_components_kg_m3", "Total mass of all components (kg/m³)"),
]


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def flatten_dict(d, parent_key=""):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items


def format_mix_output(result_dict):
    flat = flatten_dict(result_dict)
    rows = []
    seen = set()

    for key, label in DISPLAY_ORDER:
        if key in flat:
            val = flat[key]
            num = safe_float(val)
            rows.append({
                "Item": label,
                "Value": round(num, 3) if num is not None else val,
            })
            seen.add(key)

    for key, val in flat.items():
        if key in seen or key.endswith("wb_override"):
            continue
        num = safe_float(val)
        rows.append({
            "Item": key,
            "Value": round(num, 3) if num is not None else val,
        })

    return pd.DataFrame(rows)


def load_models():
    df = build_df()
    models = build_models(df)
    return df, models


class LCCOfflineApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1200x760")
        self.root.minsize(1000, 680)

        self.df, self.models = load_models()
        self.result_df = None
        self.last_summary_text = ""
        self.last_strength_df = None

        self._build_ui()
        self._update_suggested_wb()

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 10))

        ttk.Label(
            header,
            text=APP_TITLE,
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="Offline version for research and preliminary mix-design estimation.",
        ).pack(anchor="w", pady=(2, 0))

        inputs_frame = ttk.LabelFrame(outer, text="Design Inputs", padding=12)
        inputs_frame.pack(fill="x", pady=(0, 10))

        self.family_var = tk.StringVar(value=FAMILY_OPTIONS[0])
        self.age_var = tk.StringVar(value="7 Day")
        self.early_var = tk.StringVar(value="30")
        self.f28_var = tk.StringVar(value="50")
        self.use_wb_override_var = tk.BooleanVar(value=False)
        self.wb_override_var = tk.StringVar(value="")

        ttk.Label(inputs_frame, text="Concrete family").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        family_box = ttk.Combobox(
            inputs_frame,
            textvariable=self.family_var,
            values=FAMILY_OPTIONS,
            state="readonly",
            width=14,
        )
        family_box.grid(row=0, column=1, sticky="w", pady=4)
        family_box.bind("<<ComboboxSelected>>", lambda event: self._update_suggested_wb())

        ttk.Label(inputs_frame, text="Required early-age strength").grid(row=0, column=2, sticky="w", padx=(20, 8), pady=4)
        age_box = ttk.Combobox(
            inputs_frame,
            textvariable=self.age_var,
            values=list(AGE_MAP.keys()),
            state="readonly",
            width=14,
        )
        age_box.grid(row=0, column=3, sticky="w", pady=4)

        ttk.Label(inputs_frame, text="Minimum required early-age strength (MPa)").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(inputs_frame, textvariable=self.early_var, width=18).grid(row=1, column=1, sticky="w", pady=4)

        ttk.Label(inputs_frame, text="Minimum required 28-day strength (MPa)").grid(row=1, column=2, sticky="w", padx=(20, 8), pady=4)
        f28_entry = ttk.Entry(inputs_frame, textvariable=self.f28_var, width=18)
        f28_entry.grid(row=1, column=3, sticky="w", pady=4)
        f28_entry.bind("<FocusOut>", lambda event: self._update_suggested_wb())

        ttk.Checkbutton(
            inputs_frame,
            text="Override water–binder ratio",
            variable=self.use_wb_override_var,
            command=self._toggle_wb_override,
        ).grid(row=2, column=0, sticky="w", pady=4)

        ttk.Label(inputs_frame, text="Water–binder ratio override").grid(row=2, column=2, sticky="w", padx=(20, 8), pady=4)
        self.wb_entry = ttk.Entry(inputs_frame, textvariable=self.wb_override_var, width=18, state="disabled")
        self.wb_entry.grid(row=2, column=3, sticky="w", pady=4)

        self.suggested_wb_label = ttk.Label(inputs_frame, text="Suggested w/b from 28-day model: N/A")
        self.suggested_wb_label.grid(row=3, column=0, columnspan=4, sticky="w", pady=(2, 6))

        button_row = ttk.Frame(inputs_frame)
        button_row.grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))
        ttk.Button(button_row, text="Generate Mix Design", command=self.generate).pack(side="left", padx=(0, 8))
        ttk.Button(button_row, text="Save Mix Design CSV", command=self.save_mix_csv).pack(side="left", padx=(0, 8))
        ttk.Button(button_row, text="Save Strengths CSV", command=self.save_strengths_csv).pack(side="left")

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True)

        self.summary_tab = ttk.Frame(notebook, padding=10)
        self.mix_tab = ttk.Frame(notebook, padding=10)
        self.strength_tab = ttk.Frame(notebook, padding=10)

        notebook.add(self.summary_tab, text="Summary")
        notebook.add(self.mix_tab, text="Mix Design")
        notebook.add(self.strength_tab, text="Strengths")

        self.summary_text = tk.Text(self.summary_tab, wrap="word", height=16)
        self.summary_text.pack(fill="both", expand=True)
        self.summary_text.insert("1.0", "Enter the design inputs above and click Generate Mix Design.")
        self.summary_text.config(state="disabled")

        self.mix_tree = self._create_tree(self.mix_tab, ("Item", "Value"), (700, 220))
        self.strength_tree = self._create_tree(self.strength_tab, ("Age", "Predicted strength (MPa)"), (260, 220))

    def _create_tree(self, parent, columns, widths):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)

        tree = ttk.Treeview(frame, columns=columns, show="headings")
        for col, width in zip(columns, widths):
            tree.heading(col, text=col)
            tree.column(col, width=width, anchor="center" if col != columns[0] else "w")

        y_scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return tree

    def _toggle_wb_override(self):
        if self.use_wb_override_var.get():
            self.wb_entry.config(state="normal")
            suggested = self._get_suggested_wb()
            if suggested is not None and not self.wb_override_var.get().strip():
                self.wb_override_var.set(f"{suggested:.3f}")
        else:
            self.wb_entry.config(state="disabled")
            self.wb_override_var.set("")

    def _get_suggested_wb(self):
        try:
            family = self.family_var.get().strip()
            f28_min = float(self.f28_var.get())
            return float(predict_wb_from_f28_curve(self.models, f28_min, family))
        except Exception:
            return None

    def _update_suggested_wb(self):
        suggested = self._get_suggested_wb()
        if suggested is None:
            self.suggested_wb_label.config(text="Suggested w/b from 28-day model: N/A")
        else:
            self.suggested_wb_label.config(text=f"Suggested w/b from 28-day model: {suggested:.3f}")
            if self.use_wb_override_var.get() and not self.wb_override_var.get().strip():
                self.wb_override_var.set(f"{suggested:.3f}")

    def _clear_tree(self, tree):
        for row in tree.get_children():
            tree.delete(row)

    def _set_summary_text(self, text):
        self.summary_text.config(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", text)
        self.summary_text.config(state="disabled")

    def generate(self):
        try:
            family = self.family_var.get().strip()
            early_age_days = AGE_MAP[self.age_var.get()]
            early_strength = float(self.early_var.get())
            f28_min = float(self.f28_var.get())

            wb_override = None
            if self.use_wb_override_var.get():
                raw = self.wb_override_var.get().strip()
                if not raw:
                    raise ValueError("Please enter a water–binder ratio override.")
                wb_override = float(raw)

            result = design_mix_from_strengths_min(
                models=self.models,
                early_min=early_strength,
                f28_min=f28_min,
                binder_family_key=family,
                early_age_days=int(early_age_days),
                wb_override=wb_override,
            )

            pp = result["predicted_parameters"]
            wb = float(pp["water_binder_ratio"])
            water = float(pp["water_kg_m3"])
            binder_total = float(pp["binder_total_kg_m3"])
            ec_total = float(result["embodied_carbon"]["EC_total"])

            pred_f3 = float(predict_f3_from_wb(self.models, wb, family))
            pred_f7 = float(predict_f7_from_wb(self.models, wb, family))
            pred_f14 = float(predict_f14_from_wb(self.models, wb, family))
            pred_f28 = float(implied_f28_from_wb(self.models, wb, family))

            self.result_df = format_mix_output(result)
            self.last_strength_df = pd.DataFrame(
                {
                    "Age": ["3 Day", "7 Day", "14 Day", "28 Day"],
                    "Predicted strength (MPa)": [
                        round(pred_f3, 2),
                        round(pred_f7, 2),
                        round(pred_f14, 2),
                        round(pred_f28, 2),
                    ],
                }
            )

            self._clear_tree(self.mix_tree)
            for _, row in self.result_df.iterrows():
                self.mix_tree.insert("", "end", values=(row["Item"], row["Value"]))

            self._clear_tree(self.strength_tree)
            for _, row in self.last_strength_df.iterrows():
                self.strength_tree.insert("", "end", values=(row["Age"], row["Predicted strength (MPa)"]))

            summary = (
                f"Concrete family: {family}\n"
                f"Required early-age strength: ≥ {early_strength:.2f} MPa at {early_age_days} days\n"
                f"Required 28-day strength: ≥ {f28_min:.2f} MPa\n\n"
                f"Predicted water–binder ratio: {wb:.3f}\n"
                f"Predicted water: {water:.1f} kg/m³\n"
                f"Predicted total binder: {binder_total:.1f} kg/m³\n"
                f"Predicted embodied carbon: {ec_total:.1f} kgCO₂-e/m³\n\n"
                f"Predicted strengths:\n"
                f"• 3 Day:  {pred_f3:.2f} MPa\n"
                f"• 7 Day:  {pred_f7:.2f} MPa\n"
                f"• 14 Day: {pred_f14:.2f} MPa\n"
                f"• 28 Day: {pred_f28:.2f} MPa"
            )
            self.last_summary_text = summary
            self._set_summary_text(summary)
            messagebox.showinfo("Success", "Mix design generated successfully.")

        except Exception as e:
            messagebox.showerror("Error", f"The mix design could not be generated.\n\n{e}")

    def save_mix_csv(self):
        if self.result_df is None or self.result_df.empty:
            messagebox.showwarning("No data", "Please generate a mix design first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save Mix Design CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="mix_design_output.csv",
        )
        if not path:
            return

        try:
            self.result_df.to_csv(path, index=False)
            messagebox.showinfo("Saved", f"Mix design CSV saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save CSV.\n\n{e}")

    def save_strengths_csv(self):
        if self.last_strength_df is None or self.last_strength_df.empty:
            messagebox.showwarning("No data", "Please generate a mix design first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save Strengths CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="predicted_strengths.csv",
        )
        if not path:
            return

        try:
            self.last_strength_df.to_csv(path, index=False)
            messagebox.showinfo("Saved", f"Strengths CSV saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save CSV.\n\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.iconname("LCC Offline")
    except Exception:
        pass
    app = LCCOfflineApp(root)
    root.mainloop()
