#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 1.0-alpha Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""
import sys
from pathlib import Path

VAME_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(VAME_ROOT))
import vame
import matplotlib
matplotlib.use("TkAgg")  # use the TkAgg backend for interactive plots
from vame.analysis.tree_hierarchy import auto_cut_tree
from vame.analysis.community_analysis import create_community_bag
import numpy as np
from joblib import load

import os
import re
import pandas as pd
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


print(vame.__file__)


video_directory = "path/to/videos"

def get_video_paths(video_directory):
    video_paths = []
    for root, _, files in os.walk(video_directory):
        for file in files:
            video_paths.append(os.path.join(root, file))
    return video_paths

            
# Set these paths for your dataset before running.
working_directory = 'path/to/working/directory'
project = 'Model'

videos = get_video_paths(video_directory)
    
# Initialize your project
# Step 1.1:
#config = vame.init_new_project(project=project, videos=videos, working_directory=working_directory, videotype='.avi')

# After the inital creation of your project you can always access the config.yaml file 
# via specifying the path to your project
config = 'path/to/your/project/config.yaml'

# As our config.yaml is sometimes still changing a little due to updates, we have here a small function
# to update your config.yaml to the current state. Be aware that this will overwrite your current config.yaml
# and make sure to back up your version if you did parameter changes!
vame.update_config(config)

# Step 1.2:
# Align your behavior videos egocentric and create training dataset:
# pose_ref_index: list of reference coordinate indices for alignment
# Example: 0: snout, 1: forehand_left, 2: forehand_right, 3: hindleft, 4: hindright, 5: tail
vame.egocentric_alignment(config, pose_ref_index=[0,5])
# If your experiment is by design egocentrical (e.g. head-fixed experiment on treadmill etc) 
# you can use the following to convert your .csv to a .npy array, ready to train vame on it
vame.csv_to_numpy(config)

# Step 1.3:
# create the training set for the VAME model
vame.create_trainset(config, check_parameter=False)

# Step 2:
# Train VAME:
vame.train_model(config)

# Step 3:
# Evaluate model
vame.evaluate_model(config)

# Step 4:
# Segment motifs/pose
vame.pose_segmentation(config)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# The following are optional choices to create motif videos, communities/hierarchies of behavior,
# community videos

# OPTIONIAL: Create motif videos to get insights about the fine grained poses
vame.motif_videos(config, videoType='.mp4')

# OPTIONAL: Create behavioural hierarchies via community detection
vame.community(config, show_umap=True, cut_tree=None)

# --- Run the regular community detection first ---
communities_all, trees, valid_files = vame.community(config, show_umap=True, cut_tree=1)

# --- Then apply automatic cutline selection to those trees ---
DEPTH_RANGE = (1, 5)
MIN_GROUPS  = 4
MAX_GROUPS  = 10
JUMP_THRESH = 0.35

auto_cutline_per_mouse = []
auto_comms_per_mouse   = []
diagnostics_per_mouse  = []

for i, T in enumerate(trees):
    mouse_id = valid_files[i] if i < len(valid_files) else f"mouse_{i}"
    cutline, comms, info = auto_cut_tree(
        T,
        depth_range=DEPTH_RANGE,
        min_groups=MIN_GROUPS,
        max_groups=MAX_GROUPS,
        jump_frac_threshold=JUMP_THRESH,
        prefer_traverse=True,
        return_diagnostics=True,
    )

    auto_cutline_per_mouse.append(cutline)
    auto_comms_per_mouse.append(comms)
    diagnostics_per_mouse.append(info)

    print(f"[auto-cut] {mouse_id}: cutline={cutline}, groups={len(comms)}")

# Replace communities_all with auto-cut results
communities_all = auto_comms_per_mouse



# OPTIONAL: Create community videos to get insights about behavior on a hierarchical scale
vame.community_videos(config)

# OPTIONAL: Down projection of latent vectors and visualization via UMAP
vame.visualization(config, label="motif") #options: label: None, "motif", "community"
vame.visualization(config, label="None")
vame.visualization(config, label="Community")
