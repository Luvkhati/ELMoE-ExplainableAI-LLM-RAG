# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 08:41:55 2025

@author: arnab
"""

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.graph_objects as go

def create_dot_plot_plotly(
    y1_values, label1,
    y2_values, label2,
    y3_values, label3,
    title="Dot Plot",
    x_label="X-Axis",
    y_label="Y-Axis",
    save_path=None,
    tol=0.0  # set >0 for float-tolerant equality, e.g., 1e-9
):
    if len(y1_values) != len(y2_values):
        raise ValueError("y1_values and y2_values must have the same length.")

    indices = list(range(len(y1_values)))

    # Where they are (approximately) equal
    equal_indices = [
        i for i, (a, b) in enumerate(zip(y1_values, y2_values))
        if abs(a - b) <= tol
    ]
    equal_values = [y1_values[i] for i in equal_indices]

    fig = go.Figure()

    # Base series (all points shown)
    fig.add_trace(go.Scatter(
        x=indices, y=y1_values,
        mode="markers",
        name=label1,
        marker=dict(size=10, opacity=0.8),
        hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra>{label1}</extra>"
    ))

    fig.add_trace(go.Scatter(
        x=indices, y=y2_values,
        mode="markers",
        name=label2,
        marker=dict(size=10, opacity=0.8),
        hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra>{label2}</extra>"
    ))
    
    # Base series (all points shown)
    fig.add_trace(go.Scatter(
        x=indices, y=y3_values,
        mode="markers",
        name=label3,
        marker=dict(size=10, opacity=0.8),
        hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra>{label1}</extra>"
    ))


    # Highlight overlay: ring/halo around equal points
    if equal_indices:
        fig.add_trace(go.Scatter(
            x=equal_indices, y=equal_values,
            mode="markers",
            name="Equal (highlight)",
            showlegend=True,  # set to False if you don't want a legend entry
            marker=dict(
                size=16,                 # slightly bigger than base markers
                color="rgba(0,0,0,0)",   # transparent fill
                line=dict(color="black", width=2)  # ring/outline
            ),
            hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra>Equal y1 & y2</extra>"
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend=dict(title=None),
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")

    if save_path!='None':
        if save_path.lower().endswith(".html"):
            fig.write_html(save_path, include_plotlyjs="cdn", full_html=True)
        elif save_path.lower().endswith((".png", ".jpg", ".jpeg")):
            fig.write_image(save_path)  # needs kaleido
        else:
            raise ValueError("save_path must end with .html, .png, .jpg, or .jpeg")

    fig.show()

    

import numpy as np
import matplotlib.pyplot as plt

def create_dot_plot(y1_values, label1,
                    y2_values, label2,
                    y3_values, label3,
                    title="Dot Plot", x_label="X-Axis", y_label="Y-Axis",
                    save_path=None, tol=0.0):
    """
    Creates a dot plot from up to three series.
    - Plots y1, y2, y3 against shared x indices.
    - Handles unequal lengths by padding with NaNs (Matplotlib ignores them).
    - Highlights points where y1 == y2 (within tolerance) with a ring marker.
    """

    # --- sanitize/flatten to 1D float arrays; raise if ragged ---
    def to_1d_float(arr, name):
        a = np.asarray(arr, dtype=float)
        if a.ndim != 1:
            raise ValueError(f"{name} must be a 1D sequence (got shape {a.shape}).")
        return a

    y1 = to_1d_float(y1_values, "y1_values")
    y2 = to_1d_float(y2_values, "y2_values")
    y3 = to_1d_float(y3_values, "y3_values")

    # --- build common x index, pad shorter arrays with NaNs ---
    n1, n2, n3 = len(y1), len(y2), len(y3)
    M = max(n1, n2, n3)
    indices = np.arange(M)

    def pad(a, M):
        out = np.full(M, np.nan)
        out[:len(a)] = a
        return out

    y1p, y2p, y3p = pad(y1, M), pad(y2, M), pad(y3, M)

    # --- plotting ---
    plt.figure(figsize=(20, 6))

    colors = {
        label1: "blue",
        label2: "red",
        label3: "green"
    }

    plt.scatter(indices, y1p, color=colors.get(label1, "blue"),  s=100, alpha=0.7, label=label1)
    plt.scatter(indices, y2p, color=colors.get(label2, "red"),   s=100, alpha=0.7, label=label2)
    plt.scatter(indices, y3p, color=colors.get(label3, "green"), s=100, alpha=0.7, label=label3)

    # --- highlight equal points (only where both exist) ---
    min_len = min(n1, n2)
    if min_len > 0:
        equal_indices = [i for i in range(min_len) if abs(y1[i] - y2[i]) <= tol]
        if equal_indices:
            equal_values = [y1[i] for i in equal_indices]
            plt.scatter(equal_indices, equal_values,
                        facecolors='none', edgecolors='black',
                        s=300, linewidths=2, label=f"Equal {label1} & {label2}")

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    
    if save_path:
        plt.savefig(save_path, dpi=300)
        
    plt.show()





def single_dot_plot(y1_values,title="MAE Plot", x_label="samples", y_label="MAE"):
    

    plt.figure(figsize=(20, 6))
    indices = range(len(y1_values))
    plt.scatter(indices, y1_values, color='blue', s=100, alpha=0.7)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
