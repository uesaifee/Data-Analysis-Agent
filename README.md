# Data-Analysis-Agent
The agent operates on compressed schema metadata and structured analysis history, dramatically reducing computation cost and improving scalability.

# ============================================================
# ============================================================

import pandas as pd
import numpy as np
import hashlib
from scipy.stats import entropy
from typing import Dict, Any, List


# ============================================================
# Utility Functions
# ============================================================

def fingerprint_column(series: pd.Series, sample_size: int = 100) -> str:
    """Create a stable fingerprint of a column distribution."""
    series = series.dropna()
    if len(series) == 0:
        return "empty"

    sample = series.sample(min(sample_size, len(series)), random_state=42)
    return hashlib.md5(str(sample.values).encode()).hexdigest()


def is_near_constant(series: pd.Series, threshold: float = 0.98) -> bool:
    """Detect near-constant columns."""
    if series.nunique() <= 1:
        return True
    top_freq = series.value_counts(normalize=True).iloc[0]
    return top_freq >= threshold


def is_high_cardinality(series: pd.Series, threshold_ratio: float = 0.5) -> bool:
    """Detect ID-like high-cardinality columns."""
    return series.nunique() / len(series) > threshold_ratio


# ============================================================
# Schema Compressor
# ============================================================

class SchemaCompressor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def compress(self) -> Dict[str, Any]:
        schema_summary = {}

        for col in self.df.columns:
            series = self.df[col]

            col_summary = {
                "dtype": str(series.dtype),
                "missing_pct": float(series.isna().mean()),
                "n_unique": int(series.nunique()),
                "fingerprint": fingerprint_column(series)
            }

            if pd.api.types.is_numeric_dtype(series):
                col_summary.update({
                    "mean": float(series.mean()) if series.notna().any() else 0.0,
                    "std": float(series.std()) if series.notna().any() else 0.0,
                    "min": float(series.min()) if series.notna().any() else 0.0,
                    "max": float(series.max()) if series.notna().any() else 0.0
                })
            else:
                top_vals = series.value_counts(normalize=True).head(5)
                col_summary.update({
                    "top_categories": top_vals.to_dict(),
                    "entropy": float(entropy(top_vals)) if len(top_vals) > 1 else 0.0
                })

            schema_summary[col] = col_summary

        return schema_summary


# ============================================================
# History Compressor
# ============================================================

class HistoryCompressor:
    def __init__(self, max_steps: int = 10):
        self.history = []
        self.max_steps = max_steps

    def add_step(self, action: str, columns: List[str], result_summary: str):
        compressed = {
            "action": action,
            "columns": columns,
            "summary": result_summary[:300]
        }

        self.history.append(compressed)

        if len(self.history) > self.max_steps:
            self.history = self.history[-self.max_steps:]

    def summarize(self) -> Dict[str, Any]:
        return {
            "recent_steps": self.history,
            "total_steps": len(self.history)
        }


# ============================================================
# Execution Engine
# ============================================================

class ExecutionEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def execute(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        action = action_dict["action"]
        cols = action_dict.get("columns", [])

        if action == "correlation_analysis":
            result = self.df[cols].corr().to_dict()

        elif action == "groupby_mean":
            result = self.df.groupby(cols[0])[cols[1]].mean().to_dict()

        elif action == "missing_analysis":
            result = self.df[cols].isna().mean().to_dict()

        elif action == "outlier_detection":
            result = {}
            for col in cols:
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
                result[col] = {
                    "outlier_count": int(outliers),
                    "outlier_pct": float(outliers / len(self.df))
                }

        else:
            result = {"error": "Unknown action"}

        return result


# ============================================================
# Rule-Based EDA Planner (Token-Free)
# ============================================================

class EDAPlanner:
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema

    def plan(self) -> List[Dict[str, Any]]:
        actions = []

        numeric_cols = [
            col for col, meta in self.schema.items()
            if "float" in meta["dtype"] or "int" in meta["dtype"]
        ]

        categorical_cols = [
            col for col, meta in self.schema.items()
            if "object" in meta["dtype"] or "category" in meta["dtype"]
        ]

        # Missing analysis
        missing_cols = [
            col for col, meta in self.schema.items()
            if meta["missing_pct"] > 0.05
        ]
        if missing_cols:
            actions.append({
                "action": "missing_analysis",
                "columns": missing_cols
            })

        # Correlation analysis
        if len(numeric_cols) >= 2:
            actions.append({
                "action": "correlation_analysis",
                "columns": numeric_cols
            })

        # Outlier detection
        if numeric_cols:
            actions.append({
                "action": "outlier_detection",
                "columns": numeric_cols
            })

        # Groupby analysis
        if numeric_cols and categorical_cols:
            actions.append({
                "action": "groupby_mean",
                "columns": [categorical_cols[0], numeric_cols[0]]
            })

        return actions


# ============================================================
# Main Data Science Agent
# ============================================================

class DataScienceAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        # Prune useless columns before anything else
        self._prune_columns()

        self.schema = SchemaCompressor(self.df).compress()
        self.memory = HistoryCompressor()
        self.executor = ExecutionEngine(self.df)
        self.planner = EDAPlanner(self.schema)

    def _prune_columns(self):
        cols_to_drop = []

        for col in self.df.columns:
            series = self.df[col]
            if is_near_constant(series):
                cols_to_drop.append(col)
            elif is_high_cardinality(series):
                cols_to_drop.append(col)

        self.df.drop(columns=cols_to_drop, inplace=True)

    def run(self):
        actions = self.planner.plan()

        results = {}

        for action in actions:
            result = self.executor.execute(action)

            summary = str(result)[:500]
            self.memory.add_step(
                action=action["action"],
                columns=action["columns"],
                result_summary=summary
            )

            results[action["action"]] = result

        return {
            "schema": self.schema,
            "results": results,
            "history": self.memory.summarize()
        }


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    # Example dataset
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, 1000),
        "income": np.random.normal(50000, 15000, 1000),
        "gender": np.random.choice(["Male", "Female"], 1000),
        "constant_col": 1,
        "id_col": np.arange(1000)
    })

    agent = DataScienceAgent(df)
    output = agent.run()

    print("Compressed Schema:")
    print(output["schema"])

    print("\nEDA Results:")
    print(output["results"])

    print("\nCompressed History:")
    print(output["history"])
