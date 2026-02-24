<div align="center">
  
# Navigating The Unknown! (P4)
### Hands-On Aerial Robotics [RBE595]

[![Perception](https://img.shields.io/badge/Perception-Unknown%20Geometry-orange.svg)](#)
[![Simulation](https://img.shields.io/badge/Simulation-VizFlyt-blue.svg)](#)
[![Algorithm](https://img.shields.io/badge/Algorithm-Active%20Sensing-green.svg)](#)

*End-to-end autonomy pipeline designed to segment, track, and fly an autonomous quadrotor through entirely unknown and irregular 3D window geometries.*

</div>

---

## 📖 Overview

This repository holds Project 4 (P4) of the *Hands-On Aerial Robotics* course. Pushing boundaries beyond traditional racing constraints, this project focuses on **Navigating The Unknown**—flying a quadrotor through *irregular, dynamic, and completely unseen window shapes* (e.g., for search and rescue operations or exploration).

Unlike prior projects where the drone expects a square frame with standard checkerboard corners, the perception logic in P4 cannot assume any prior topological features. The window textures may blend completely with the background, missing color cues and forcing reliance on advanced active sensing and depth dynamics to find the gap.

### Key Algorithmic Innovations
1. **Geometry-Agnostic Perception:** Overcoming the limitations of standard bounding-box detectors. This pipeline uses dense sensing techniques (optical flow, depth thresholds, or active camera maneuvers) to segment convex topological openings in highly noisy environments without requiring a pre-defined CNN structure trained on specific window masks.
2. **Dynamic Gap Identification:** Filtering the largest reliable aperture from false-positive holes on textured environments running at >5Hz on the simulated Orin Nano CPU.
3. **Reactive 3D Planning:** Translating the dynamically morphing gap centroid into viable 3D waypoints for the hierarchical planner (RRT* mapping).
4. **VizFlyt Control execution:** Robust execution of the generated splines utilizing cascaded positioning controllers matching exact velocity demands in real-time.

---

## 🚀 Execution & Methodology

### 1. Robust Dataset & Validation (Blender)
Traditional datasets fall short for evaluating "unknowns". The provided `Data_Generation` scripts leverage the full Blender pipeline to systematically test the algorithm against bizarre window shapes—rendered dynamically using alpha masking and extreme texture camouflage (Foreground matching Background).

### 2. Live Gap Inference & Traversal
Upon spawning, the drone's first task is often active perception. To disambiguate background walls from the cutout, the quadrotor performs subtle exploratory movements. By tracking the differential shifts (or analyzing the depth discontinuities across the planar foamcore), the algorithm identifies the convex bounds of the largest continuous gap and commits to a flight trajectory straight through the center.

---

## 🎥 Simulation Showcase

<div align="center">
  <img src="assets/Video.gif" width="65%" alt="Autonomous Navigation Flight">
</div>

*Visualizations of the quadrotor discovering the unknown gap profile and flying through it cleanly.*

---

## 🚀 Quick Start & Usage

### Requirements
```bash
pip install numpy scipy matplotlib 
# Depending on custom perception stack: torch, torchvision, opencv-python
```

### Usage
```bash
# Execute the primary navigation and exploration node
cd src/
python main.py
```

*Note: The Blender script files for experimenting with the custom Alpha windows are located in the `Data_Generation/Blender/` directory.*

---

## 📁 Repository Structure

```text
RBE595-Navigating-The-Unknown-P4/
├── Data_Generation/       # Blender testing assets for spawning randomized window geometries
├── src/                   # Main autonomy stack (Gap Search, Waypoint Control, VizFlyt)
├── Report.pdf             # Exhaustive analysis of the active sensing algorithms executed
├── assets/                # Rendered animations of successful gap traversals
└── README.md              # Project overview
```
