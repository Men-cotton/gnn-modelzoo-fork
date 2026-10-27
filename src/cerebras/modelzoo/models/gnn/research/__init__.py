"""Experimental GraphSAGE representations using the existing GNN runtime."""

from pathlib import Path

GNN = Path(__file__).resolve().parents[1]
ROOT = GNN.parents[4]
DATA = GNN / "data/datasets"
