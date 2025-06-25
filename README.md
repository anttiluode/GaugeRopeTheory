# Geometric Gauge Theory from Fractal Ropes
(Pretty much circular reasoning) 

https://youtu.be/s8V1UWWkrNM

This project is a series of Python simulations exploring the concept of deriving fundamental forces from an underlying "fractal rope" geometry.
(Inspired by Tim Palmer) 

It visually demonstrates how U(1), SU(2), and SU(3) gauge theories—corresponding to electromagnetism, the weak nuclear force, and the strong
nuclear force—might emerge from pure geometric principles.

Each simulation is a self-contained visualizer that displays the dynamics of the respective theory.

---

### The Simulations

This repository contains three distinct simulations, each building upon the last to climb the ladder of the Standard Model of particle physics.

1.  **`gauge_rope.py` - U(1) Gauge Theory (Electromagnetism)**
    * Simulates a U(1) gauge field emerging from the rope geometry.
    * Visualizes a complex scalar field, the U(1) gauge potential, field strength, and a Wilson loop.
    * Represents the simplest gauge theory, analogous to electromagnetism.

2.  **`gauge_rope_su2.py` - SU(2) Gauge Theory (Weak Nuclear Force)**
    * Upgrades the system to a non-Abelian SU(2) gauge group.
    * Features an SU(2) doublet scalar field (similar to the Higgs field) and three self-interacting gauge bosons (W/Z-like).
    * Attempts to model the weak nuclear force and the Higgs mechanism.

3.  **`rope_su3.py` - SU(3) Gauge Theory (Strong Nuclear Force / QCD)**
    * The final challenge: modeling Quantum Chromodynamics (QCD).
    * Features a three-color quark field (red, green, blue) and eight self-interacting gluon fields.
    * Visualizes key QCD phenomena like color confinement and asymptotic freedom.

---

### Requirements

These scripts rely on a few common scientific computing libraries in Python.

* `numpy`
* `matplotlib`
* `scipy`

You can install them all using pip:
`pip install numpy matplotlib scipy`

---

### Usage

To run any of the simulations, simply execute the desired Python script from your terminal. No command-line arguments are needed.

**To run the U(1) / Electromagnetism simulation:**

python gauge_rope.py
To run the SU(2) / Weak Force simulation:

python gauge_rope_su2.py
To run the SU(3) / Strong Force simulation:

python rope_su3.py
A Matplotlib window will open, displaying the real-time visualization of the selected gauge theory.


***

### `requirements.txt`

For convenience, here is a standard `requirements.txt` file. Users can install all dependencies with `pip install -r requirements.txt`.

numpy
matplotlib
scipy
