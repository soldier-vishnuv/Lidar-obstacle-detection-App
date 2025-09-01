import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os
from sklearn.cluster import DBSCAN
from typing import List, Tuple
from io import StringIO
from collections import defaultdict

# Page config
st.set_page_config(page_title="LIDAR Obstacle Detection App", layout="wide")

# Sidebar: Mode selection
st.sidebar.title("LIDAR Obstacle Detection App")
mode = st.sidebar.radio("Select Input Mode", ["Use Test Case", "Upload Your Own File"])
uploaded_file = None

# Handle test case files
if mode == "Use Test Case":
    test_dir = "Test_Case_CSV_Files"
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".csv")])
    selected_file = st.sidebar.selectbox("Choose Test Case", ["-- Select a test case --"] + test_files)

    if selected_file != "-- Select a test case --":
        with open(os.path.join(test_dir, selected_file), "r") as f:
            uploaded_file = StringIO(f.read())
        st.sidebar.success(f"Loaded: {selected_file}")
    else:
        st.sidebar.info("Please select a test case file to proceed.")

# Handle custom upload
elif mode == "Upload Your Own File":
    uploaded_file = st.sidebar.file_uploader("Upload LIDAR CSV File", type="csv")

# Title
st.title("LIDAR Obstacle Detection App")

# Obstacle detection logic
def find_obstacles(points: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float, float]]:
    return [
        (float(x), float(y), float(z), math.hypot(float(x), float(y)))
        for x, y, z in points
        if math.hypot(float(x), float(y)) < 10
    ]

def categorize_danger(dist):
    if dist < 3:
        return 'High'
    elif dist < 6:
        return 'Medium'
    else:
        return 'Low'

def timed_obstacle_detection(points):
    start = time.time()
    result = find_obstacles(points)
    end = time.time()
    return result, round((end - start) * 1000, 2)

# File processing
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, header=None, names=["x", "y", "z"])
        points = list(df.itertuples(index=False, name=None))
        obstacles, exec_time = timed_obstacle_detection(points)

        if not obstacles:
            st.warning("No obstacles detected within 10 meters.")
        else:
            st.success(f"{len(obstacles)} obstacle(s) detected in {exec_time} ms")

            obs_df = pd.DataFrame(obstacles, columns=["x", "y", "z", "distance"])
            obs_df["danger"] = obs_df["distance"].apply(categorize_danger)

            st.markdown("### Detected Obstacles with Danger Level")
            st.dataframe(obs_df.style.background_gradient(cmap="coolwarm", subset=["distance"]))

            # Clustering
            clustering = DBSCAN(eps=1.2, min_samples=2).fit(obs_df[["x", "y"]])
            obs_df["cluster"] = clustering.labels_
            n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            st.markdown(f"### {n_clusters} Obstacle Cluster(s) Identified")

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 8))
            danger_colors = {"High": "red", "Medium": "orange", "Low": "green"}

            for danger_level in ["High", "Medium", "Low"]:
                danger_points = obs_df[obs_df["danger"] == danger_level]
                ax.scatter(danger_points["x"], danger_points["y"],
                           s=100, alpha=0.8, edgecolors='black',
                           c=danger_points["danger"].map(danger_colors),
                           label=f'{danger_level} Risk ({len(danger_points)})')

            # Draw clusters
            for cl in obs_df["cluster"].unique():
                if cl == -1:
                    continue
                cluster = obs_df[obs_df["cluster"] == cl]
                cx, cy = cluster["x"].mean(), cluster["y"].mean()
                ax.text(cx, cy, f"Cluster {cl}", fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.6))

            # Range & origin
            ax.scatter(0, 0, color='blue', label='Robot Origin')
            ax.add_patch(plt.Circle((0, 0), 10, fill=False, linestyle='--', edgecolor='blue', linewidth=1))
            ax.set_aspect('equal')
            ax.set_xlabel("X (meters)")
            ax.set_ylabel("Y (meters)")
            ax.legend()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file or choose a test case to begin.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;">
        <strong>Developed by Team-Mind_Mesh</strong><br>
        Developers: Amrutha D , Vishnu V<br>
        College: REVA University, Bangalore<br>
    </div>
    """,
    unsafe_allow_html=True
)
# Hide Streamlit footer
st.markdown("""
    <style>
        footer {visibility: hidden;}
        .stApp {bottom: 0;}
    </style>

""", unsafe_allow_html=True)    
