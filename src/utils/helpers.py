import pickle
import numpy as np
import os
from pathlib import Path
import pandas as pd

def load_pkl(path):
    """
    Load pickle file
    """
    with open(path, 'rb') as f:
        obj = pickle.load(f)
        return obj


def find_project_root(start=None):
    current_path = Path(__file__).resolve()

    start = Path(start or Path.cwd()).resolve()
    root_markers = {".git", "pyproject.toml", "setup.py", ".root"}
    for parent in current_path.parents:
        if any((parent / marker).exists() for marker in root_markers):
            return parent
    
    raise RuntimeError("Could not find project root containing data/easdrl and src")


def df_to_latex(df, columns_to_drop=[]):
    latex_table = (
        df.drop(columns=columns_to_drop)
        .style
        .format(escape="latex")                # Escapes the data cells
        .format_index(escape="latex", axis=1)  # Escapes column names
        .format_index(escape="latex", axis=0)  # Escapes index names
        .to_latex()
    )
    return latex_table