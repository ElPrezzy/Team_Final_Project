"""
LendingClub EDA - "Current" loans analysis (OOP version)

Main hypothesis: debt_payments loans 
are more likely to stay Current than small_business loans.
Educational loans are kept as a third group for context.

Supporting hypothesis: shorter-term loans (36 months) are more likely
to stay Current than longer-term loans (60 months).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest


# ---------- stats helpers ----------
class ProportionStats:
    """Wilson CIs, diff-in-prop CIs, two-prop z-test."""

    def __init__(self, z=1.96):
        self.z = z

    def wilson_ci(self, k, n):
        if n == 0:
            return (np.nan, np.nan)
        z = self.z
        p = k / n
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
        return (center - half, center + half)

    def diff_ci(self, k1, n1, k2, n2):
        p1, p2 = k1 / n1, k2 / n2
        diff = p1 - p2
        se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        return diff, diff - self.z * se, diff + self.z * se

    @staticmethod
    def two_prop_z(k1, n1, k2, n2):
        return proportions_ztest([k1, k2], [n1, n2])


# ---------- data loader ----------
class LoanDataLoader:
    """Pulls just the columns we need from the huge CSV."""

    COLS = ["purpose", "loan_status", "issue_d", "term"]
    # filenames we'll try if the given path doesn't exist
    FALLBACK_NAMES = [
        "LendingClub_Data.csv",
        "LendingClub_Data (2).csv",
        "lendingclub_data.csv",
        "loan.csv",
    ]

    def __init__(self, csv_path):
        self.csv_path = self._resolve(csv_path)

    def _resolve(self, csv_path):
        if os.path.isfile(csv_path):
            return csv_path
        # try fallback names in the same folder as the requested path,
        # and in the script's own folder
        search_dirs = [os.path.dirname(csv_path),
                       os.path.dirname(os.path.abspath(__file__))]
        for d in search_dirs:
            for name in self.FALLBACK_NAMES:
                candidate = os.path.join(d, name)
                if os.path.isfile(candidate):
                    print(f"(note: using {candidate} instead of {csv_path})")
                    return candidate
        # nothing found - list what IS in the folder so the user can pick
        target_dir = os.path.dirname(csv_path) or "."
        try:
            csvs = [f for f in os.listdir(target_dir) if f.lower().endswith(".csv")]
        except FileNotFoundError:
            csvs = []
        msg = (f"Could not find CSV at {csv_path}.\n"
               f"CSVs in {target_dir}: {csvs or '[none]'}\n"
               f"Update CSV_PATH at the bottom of the script.")
        raise FileNotFoundError(msg)

    def load(self):
        print(f"Loading CSV (only 4 cols) from:\n  {self.csv_path}")
        df = pd.read_csv(self.csv_path, usecols=self.COLS, low_memory=False)
        print(f"Loaded {len(df):,} rows\n")
        return df


# ---------- preprocessor ----------
class LoanPreprocessor:
    """Purpose grouping + vintage filter + flags."""

    PURPOSE_MAP = {
        "debt_consolidation": "debt_payments",
        "credit_card":        "debt_payments",
        "small_business":     "investment",
        "educational":        "educational",
    }
    GROUP_ORDER = ["debt_payments", "investment", "educational"]
    VINTAGE_YEARS = [2014, 2015]

    def show_raw_value_counts(self, df):
        print("=== purpose ===")
        print(df["purpose"].value_counts(dropna=False))
        print("\n=== loan_status ===")
        print(df["loan_status"].value_counts(dropna=False))
        print("\n=== term ===")
        print(df["term"].value_counts(dropna=False))
        print()

    def apply_purpose_groups(self, df):
        df = df.copy()
        df["purpose_group"] = df["purpose"].map(self.PURPOSE_MAP)
        df = df[df["purpose_group"].notna()].copy()
        print(f"After purpose filter: {len(df):,} rows")
        print(df["purpose_group"].value_counts(), "\n")
        return df

    def filter_vintage(self, df):
        df = df.copy()
        df["issue_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y",
                                        errors="coerce")
        df["issue_year"] = df["issue_dt"].dt.year
        df = df[df["issue_year"].isin(self.VINTAGE_YEARS)].copy()
        print(f"After {self.VINTAGE_YEARS} filter: {len(df):,} rows\n")
        return df

    def add_flags(self, df):
        df = df.copy()
        df["is_current"] = (df["loan_status"] == "Current").astype(int)
        # normalize term: " 36 months" / " 60 months" -> "36 mo" / "60 mo"
        df["term_clean"] = df["term"].str.strip().str.replace(" months", " mo",
                                                              regex=False)
        return df

    def run(self, df):
        self.show_raw_value_counts(df)
        df = self.apply_purpose_groups(df)
        df = self.filter_vintage(df)
        df = self.add_flags(df)
        return df


# ---------- hypothesis testing ----------
class HypothesisTester:
    """Success rates, chi-square, pairwise z-tests for any group column."""

    def __init__(self, df, stats_util=None):
        self.df = df
        self.stats = stats_util or ProportionStats()
        self.results = {}  # keyed by analysis name

    def success_rates(self, group_col, group_order, label):
        rows = []
        for g in group_order:
            sub = self.df[self.df[group_col] == g]
            n = len(sub)
            cur = int(sub["is_current"].sum())
            rate = cur / n if n else np.nan
            lo, hi = self.stats.wilson_ci(cur, n)
            rows.append([g, n, cur, rate, lo, hi])
        summary = pd.DataFrame(rows,
            columns=["group", "n", "current_count",
                     "success_rate", "ci_lo", "ci_hi"])
        print(f"=== success rates: {label} ===")
        print(summary.to_string(index=False,
              float_format=lambda x: f"{x:.4f}"))
        print()
        self.results.setdefault(label, {})["summary"] = summary
        return summary

    def chi_square(self, group_col, group_order, label):
        ct = pd.crosstab(self.df[group_col],
                         self.df["is_current"]).loc[group_order]
        print(f"=== contingency table: {label} ===")
        print(ct, "\n")
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        n_total = ct.values.sum()
        cramers_v = np.sqrt(chi2 / (n_total * (min(ct.shape) - 1)))
        result = {"chi2": chi2, "p": p, "dof": dof,
                  "cramers_v": cramers_v, "contingency": ct}
        print(f"chi2 = {chi2:.4f}")
        print(f"p-value = {p:.4g}")
        print(f"dof = {dof}")
        print(f"Cramér's V = {cramers_v:.4f}")
        print()
        self.results.setdefault(label, {}).update(result)
        return result

    def pairwise(self, summary, pairs, label):
        rows = []
        for a, b in pairs:
            ka = int(summary.loc[summary["group"] == a, "current_count"].iloc[0])
            na = int(summary.loc[summary["group"] == a, "n"].iloc[0])
            kb = int(summary.loc[summary["group"] == b, "current_count"].iloc[0])
            nb = int(summary.loc[summary["group"] == b, "n"].iloc[0])
            zstat, pval = self.stats.two_prop_z(ka, na, kb, nb)
            diff, lo, hi = self.stats.diff_ci(ka, na, kb, nb)
            rows.append([f"{a} vs {b}", diff, lo, hi, zstat, pval])
        pairs_df = pd.DataFrame(rows,
            columns=["comparison", "diff", "ci_lo", "ci_hi",
                     "z", "p_value"])
        print(f"=== pairwise two-prop z-tests: {label} ===")
        print(pairs_df.to_string(index=False,
              float_format=lambda x: f"{x:.4f}"))
        print()
        self.results.setdefault(label, {})["pairs"] = pairs_df
        return pairs_df


# ---------- plots ----------
class LoanPlotter:
    """Two bar charts: main hypothesis + supporting hypothesis."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        sns.set_style("whitegrid")

    def _bar_with_ci(self, summary, group_order, title, filename,
                     palette, ylabel="Success rate (% Current)"):
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(group_order))
        rates = summary["success_rate"].values
        err_lo = rates - summary["ci_lo"].values
        err_hi = summary["ci_hi"].values - rates
        bars = ax.bar(x, rates, yerr=[err_lo, err_hi], capsize=6,
                      color=palette, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(group_order)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, max(rates) * 1.25)
        for bar, r in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{r:.1%}", ha="center", fontsize=10)
        plt.tight_layout()
        path = os.path.join(self.out_dir, filename)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"saved {path}")
        return path

    def main_chart(self, summary, group_order):
        return self._bar_with_ci(
            summary, group_order,
            title="Main: success rate by purpose group (2014-2015)",
            filename="success_rate_by_purpose.png",
            palette=["#4C72B0", "#DD8452", "#55A467"],
        )

    def supporting_chart(self, summary, group_order):
        return self._bar_with_ci(
            summary, group_order,
            title="Supporting: success rate by loan term (2014-2015)",
            filename="success_rate_by_term.png",
            palette=["#8172B2", "#937860"],
        )


# ---------- orchestrator ----------
class LendingClubAnalysis:
    """Ties it all together."""

    PURPOSE_PAIRS = [
        ("debt_payments", "investment"),    # main hypothesis
        ("debt_payments", "educational"),   # supporting context
        ("investment",    "educational"),   # supporting context
    ]
    TERM_ORDER = ["36 mo", "60 mo"]
    TERM_PAIRS = [("36 mo", "60 mo")]

    def __init__(self, csv_path, out_dir=None):
        self.csv_path = csv_path
        # default outputs next to the script, not next to the csv
        self.out_dir = out_dir or os.path.dirname(os.path.abspath(__file__))
        self.loader = LoanDataLoader(csv_path)
        self.prep = LoanPreprocessor()
        self.df = None
        self.tester = None
        self.plotter = None

    def run(self):
        raw = self.loader.load()
        self.df = self.prep.run(raw)

        self.tester = HypothesisTester(self.df)

        # --- main hypothesis: purpose ---
        print(">>> MAIN HYPOTHESIS: debt vs business (educational for context)")
        purpose_summary = self.tester.success_rates(
            "purpose_group", self.prep.GROUP_ORDER, "purpose")
        self.tester.chi_square(
            "purpose_group", self.prep.GROUP_ORDER, "purpose")
        self.tester.pairwise(
            purpose_summary, self.PURPOSE_PAIRS, "purpose")

        # --- supporting hypothesis: term length ---
        print(">>> SUPPORTING HYPOTHESIS: 36 mo vs 60 mo term")
        term_summary = self.tester.success_rates(
            "term_clean", self.TERM_ORDER, "term")
        self.tester.chi_square(
            "term_clean", self.TERM_ORDER, "term")
        self.tester.pairwise(
            term_summary, self.TERM_PAIRS, "term")

        # --- bonus crosstab so you can talk about the interaction in the report ---
        print(">>> purpose x term success rate (for report context) ===")
        cross = (self.df.groupby(["purpose_group", "term_clean"])["is_current"]
                        .mean().unstack().loc[self.prep.GROUP_ORDER])
        print(cross.to_string(float_format=lambda x: f"{x:.4f}"), "\n")

        # --- two charts ---
        self.plotter = LoanPlotter(self.out_dir)
        self.plotter.main_chart(purpose_summary, self.prep.GROUP_ORDER)
        self.plotter.supporting_chart(term_summary, self.TERM_ORDER)

        self._save_clean_csv()
        print("\ndone.")

    def _save_clean_csv(self, filename="lc_analysis.csv"):
        path = os.path.join(self.out_dir, filename)
        cols = ["purpose", "purpose_group", "loan_status", "is_current",
                "issue_d", "issue_year", "term", "term_clean"]
        self.df[cols].to_csv(path, index=False)
        print(f"saved {path}")


# ---------- entry point ----------
if __name__ == "__main__":
    # CSV lives in Downloads; plots + cleaned csv save next to this script
    CSV_PATH = r"C:\Users\zeusp\Downloads\LendingClub_Data.csv"
    LendingClubAnalysis(CSV_PATH).run()