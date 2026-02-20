# 🧠 Data Science Agent (Low-Token EDA)

---

## 📦 Project Structure

```
ds_agent/
│
├── agent.py
├── schema_compressor.py
├── eda_engine.py
├── memory_manager.py
├── llm_interface.py
├── token_budget.py
└── main.py
```

---

# 1️⃣ Schema Compression Module

import pandas as pd
import numpy as np

class SchemaCompressor:
    def __init__(self, top_k=5):
        self.top_k = top_k

    def compress(self, df: pd.DataFrame) -> dict:
        schema_summary = {}

        for col in df.columns:
            series = df[col]
            col_type = str(series.dtype)

            summary = {
                "type": col_type,
                "null_pct": round(series.isna().mean(), 4),
            }

            if np.issubdtype(series.dtype, np.number):
                summary.update({
                    "min": float(series.min()) if not series.isna().all() else None,
                    "max": float(series.max()) if not series.isna().all() else None,
                    "mean": float(series.mean()) if not series.isna().all() else None,
                    "std": float(series.std()) if not series.isna().all() else None,
                })

            elif series.nunique() < 50:
                top_vals = series.value_counts().head(self.top_k).index.tolist()
                summary.update({
                    "cardinality": int(series.nunique()),
                    "top_values": top_vals
                })

            schema_summary[col] = summary

        return schema_summary

    def to_compact_string(self, schema_summary: dict) -> str:
        parts = []
        for col, meta in schema_summary.items():
            if meta["type"].startswith("int") or meta["type"].startswith("float"):
                parts.append(
                    f"{col}:{meta['type']}|μ={meta.get('mean')}|σ={meta.get('std')}|null={meta['null_pct']}"
                )
            else:
                parts.append(
                    f"{col}:{meta['type']}|card={meta.get('cardinality')}|null={meta['null_pct']}"
                )

        return "\n".join(parts)
```

---

# 2️⃣ Compressed Memory Manager

class MemoryManager:
    def __init__(self):
        self.steps = []

    def add_step(self, step_type: str, details: str):
        entry = f"{step_type}:{details}"
        self.steps.append(entry)

    def get_compressed_history(self, max_steps=10):
        return "\n".join(self.steps[-max_steps:])
```

Instead of storing verbose conversation, we store structured steps like:

```
distribution:age
correlation:age,income=0.62
filter:region=EU
```

---

# 3️⃣ Automated EDA Engine

import pandas as pd
import numpy as np

class EDAEngine:
    def __init__(self, memory_manager):
        self.memory = memory_manager

    def basic_overview(self, df: pd.DataFrame):
        overview = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "missing_pct": df.isna().mean().mean()
        }
        self.memory.add_step("overview", str(overview))
        return overview

    def correlations(self, df: pd.DataFrame):
        numeric_df = df.select_dtypes(include=np.number)
        corr = numeric_df.corr()
        self.memory.add_step("correlation", "matrix_computed")
        return corr

    def distribution(self, df: pd.DataFrame, column: str):
        if column not in df.columns:
            return None
        desc = df[column].describe()
        self.memory.add_step("distribution", column)
        return desc
```

---

# 4️⃣ Token Budget Manager

class TokenBudget:
    def __init__(self, max_tokens=4000):
        self.max_tokens = max_tokens

    def estimate_tokens(self, text: str):
        return int(len(text) / 4)  # rough approximation

    def enforce_limit(self, text: str):
        if self.estimate_tokens(text) > self.max_tokens:
            return text[: self.max_tokens * 4]
        return text
```

---

# 5️⃣ LLM Interface (Token-Optimized Prompting)

from openai import OpenAI

class LLMInterface:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def ask(self, query, schema_context, history_context):
        prompt = f"""
You are a data science assistant.

Compressed Schema:
{schema_context}

Compressed History:
{history_context}

User Question:
{query}

Provide structured analytical insights.
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        return response.choices[0].message.content
```

---

# 6️⃣ Agent Orchestrator

from schema_compressor import SchemaCompressor
from memory_manager import MemoryManager
from eda_engine import EDAEngine
from token_budget import TokenBudget
from llm_interface import LLMInterface

class DataScienceAgent:
    def __init__(self, api_key):
        self.schema_compressor = SchemaCompressor()
        self.memory = MemoryManager()
        self.eda = EDAEngine(self.memory)
        self.token_budget = TokenBudget()
        self.llm = LLMInterface(api_key)
        self.schema_context = None

    def load_dataset(self, df):
        schema = self.schema_compressor.compress(df)
        self.schema_context = self.schema_compressor.to_compact_string(schema)

    def run_eda(self, df):
        self.eda.basic_overview(df)
        self.eda.correlations(df)

    def ask(self, query):
        history = self.memory.get_compressed_history()
        prompt = self.token_budget.enforce_limit(history)
        return self.llm.ask(query, self.schema_context, prompt)
