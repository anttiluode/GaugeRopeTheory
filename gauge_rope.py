"""
U(1) GAUGE THEORY FROM FRACTAL ROPE + COM-INSTANTON HYBRID
=========================================================

Combines Palmer's fractal rope with COM-instanton to derive U(1) gauge theory:

1. Complex COM-instanton field φ(x,t) = |φ|e^(iθ) with U(1) phase
2. Fractal rope geometry generates connection 1-form A_μ
3. Scout as charged particle coupled to emergent gauge field
4. Strand selection = gauge fixing mechanism
5. Emergent Lagrangian: L = |D_μφ|² - V(|φ|²) - ¼F_μν F^μν

Key Physics:
- Rope winding → Berry connection → U(1) gauge potential
- Strand coherence → gauge field strength
- Scout charge → minimal coupling to gauge field
- Measurement → gauge fixing via strand selection
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import center_of_mass
from scipy.interpolate import griddata

class U1GaugeRopeField:
    """Combined fractal rope + COM-instanton system with emergent U(1) gauge theory"""
    
    def __init__(self, grid_size=96, n_strands=8):
        self.grid_size = grid_size
        self.n_strands = n_strands
        self.dt = 0.03
        self.dx = 1.0
        self.time = 0.0
        
        # Complex COM-instanton field φ = |φ|e^(iθ)
        self.phi_mag = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.phi_phase = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.phi_mag_prev = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.phi_phase_prev = np.zeros((grid_size, grid_size), dtype=np.float64)
        
        # U(1) gauge potential A_μ = (A_t, A_x, A_y) from rope geometry
        self.A_t = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.A_x = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.A_y = np.zeros((grid_size, grid_size), dtype=np.float64)
        
        # Field strength tensor F_μν
        self.F_xy = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.F_tx = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.F_ty = np.zeros((grid_size, grid_size), dtype=np.float64)
        
        # Fractal rope strands
        self.rope_strands = []
        self.strand_phases = []
        self.strand_coherences = []
        self.selected_strand_idx = 0
        
        # Scout as charged particle
        self.scout_pos = np.array([grid_size/2, grid_size/2], dtype=float)
        self.scout_vel = np.zeros(2, dtype=float)
        self.scout_charge = 1.0  # U(1) charge
        self.scout_mass = 2.0
        
        # Coupling constants
        self.g_coupling = 0.3  # Gauge coupling
        self.lambda_potential = 0.1  # Self-interaction
        
        # Initialize system
        self.initialize_complex_instanton()
        self.create_fractal_rope_strands()
        
        # History for analysis
        self.gauge_invariant_history = []
        self.wilson_loop_history = []
        self.lagrangian_density_history = []
        
    def initialize_complex_instanton(self):
        """Initialize complex COM-instanton with U(1) phase structure"""
        center_x, center_y = self.grid_size * 0.4, self.grid_size * 0.5
        
        y_coords, x_coords = np.ogrid[:self.grid_size, :self.grid_size]
        dist_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2
        
        # Magnitude profile (soliton-like)
        self.phi_mag = 1.5 * np.exp(-dist_sq / (12**2))
        
        # Phase profile with topological winding
        self.phi_phase = np.arctan2(y_coords - center_y, x_coords - center_x)
        
        # Initialize time derivatives
        self.phi_mag_prev = self.phi_mag.copy()
        self.phi_phase_prev = self.phi_phase.copy()
        
    def create_fractal_rope_strands(self):
        """Create rope strands that generate U(1) gauge potential"""
        # Create rope centerline
        t = np.linspace(0, 4*np.pi, 100)
        rope_center = np.array([
            self.grid_size/2 + 10*np.cos(t),
            self.grid_size/2 + 10*np.sin(t),
            5*np.sin(2*t)
        ]).T
        
        # Create strands wound around centerline
        for i in range(self.n_strands):
            phase = 2*np.pi*i/self.n_strands
            
            # Helical winding with fractal sub-structure
            radius = 3 + 0.5*np.sin(8*t + phase)
            strand_x = rope_center[:, 0] + radius*np.cos(3*t + phase)
            strand_y = rope_center[:, 1] + radius*np.sin(3*t + phase)
            strand_z = rope_center[:, 2] + 0.5*np.cos(6*t + phase)
            
            strand_path = np.array([strand_x, strand_y, strand_z]).T
            self.rope_strands.append(strand_path)
            
            # Each strand carries U(1) phase
            self.strand_phases.append(phase)
            self.strand_coherences.append(1.0)
    
    def compute_gauge_potential_from_rope(self):
        """Derive U(1) gauge potential from rope geometry"""
        # Reset gauge potential
        self.A_x.fill(0)
        self.A_y.fill(0)
        self.A_t.fill(0)
        
        # Grid coordinates
        x_grid = np.arange(self.grid_size)
        y_grid = np.arange(self.grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Compute gauge potential from rope strand geometry
        for i, strand in enumerate(self.rope_strands):
            coherence = self.strand_coherences[i]
            strand_phase = self.strand_phases[i]
            
            # Project 3D strand onto 2D grid
            strand_2d = strand[:, :2]  # (x, y) coordinates
            
            # Compute Berry connection from strand geometry
            for j in range(len(strand_2d)-1):
                x1, y1 = strand_2d[j]
                x2, y2 = strand_2d[j+1]
                
                # Discrete line integral contribution
                if 0 <= x1 < self.grid_size and 0 <= y1 < self.grid_size:
                    ix1, iy1 = int(x1), int(y1)
                    
                    # Tangent vector
                    dx = x2 - x1
                    dy = y2 - y1
                    
                    # Gauge potential contribution (∇ × A gives field strength)
                    phase_gradient = np.cos(strand_phase + 0.1*j)
                    
                    # Add to gauge potential with proper coherence weighting
                    influence_radius = 8
                    for di in range(-influence_radius, influence_radius+1):
                        for dj in range(-influence_radius, influence_radius+1):
                            xi, yi = ix1 + di, iy1 + dj
                            if 0 <= xi < self.grid_size and 0 <= yi < self.grid_size:
                                dist = np.sqrt(di**2 + dj**2)
                                if dist < influence_radius:
                                    weight = coherence * np.exp(-dist**2 / (influence_radius/2)**2)
                                    
                                    self.A_x[yi, xi] += weight * phase_gradient * dx / (2*np.pi)
                                    self.A_y[yi, xi] += weight * phase_gradient * dy / (2*np.pi)
        
        # Smooth the gauge potential
        self.A_x = self.gaussian_smooth(self.A_x, sigma=1.5)
        self.A_y = self.gaussian_smooth(self.A_y, sigma=1.5)
        
        # Temporal component from field dynamics
        phi_magnitude = self.phi_mag
        self.A_t = 0.1 * self.g_coupling * phi_magnitude * np.cos(self.phi_phase)
    
    def gaussian_smooth(self, field, sigma=1.0):
        """Simple Gaussian smoothing"""
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(field, sigma=sigma, mode='wrap')
    
    def compute_field_strength(self):
        """Compute U(1) field strength tensor F_μν"""
        # F_xy = ∂_x A_y - ∂_y A_x
        dA_y_dx = (np.roll(self.A_y, -1, axis=1) - np.roll(self.A_y, 1, axis=1)) / (2*self.dx)
        dA_x_dy = (np.roll(self.A_x, -1, axis=0) - np.roll(self.A_x, 1, axis=0)) / (2*self.dx)
        self.F_xy = dA_y_dx - dA_x_dy
        
        # F_tx = ∂_t A_x - ∂_x A_t
        dA_t_dx = (np.roll(self.A_t, -1, axis=1) - np.roll(self.A_t, 1, axis=1)) / (2*self.dx)
        # Approximate ∂_t A_x using finite difference
        dA_x_dt = (self.A_x - getattr(self, 'A_x_prev', self.A_x)) / self.dt
        self.F_tx = dA_x_dt - dA_t_dx
        
        # F_ty = ∂_t A_y - ∂_y A_t  
        dA_t_dy = (np.roll(self.A_t, -1, axis=0) - np.roll(self.A_t, 1, axis=0)) / (2*self.dx)
        dA_y_dt = (self.A_y - getattr(self, 'A_y_prev', self.A_y)) / self.dt
        self.F_ty = dA_y_dt - dA_t_dy
        
        # Store previous values
        self.A_x_prev = self.A_x.copy()
        self.A_y_prev = self.A_y.copy()
    
    def covariant_derivative(self, field_mag, field_phase):
        """Compute covariant derivative D_μ φ = (∂_μ - i g A_μ) φ"""
        # Spatial derivatives
        d_phi_mag_dx = (np.roll(field_mag, -1, axis=1) - np.roll(field_mag, 1, axis=1)) / (2*self.dx)
        d_phi_mag_dy = (np.roll(field_mag, -1, axis=0) - np.roll(field_mag, 1, axis=0)) / (2*self.dx)
        
        d_phi_phase_dx = (np.roll(field_phase, -1, axis=1) - np.roll(field_phase, 1, axis=1)) / (2*self.dx)
        d_phi_phase_dy = (np.roll(field_phase, -1, axis=0) - np.roll(field_phase, 1, axis=0)) / (2*self.dx)
        
        # Covariant derivatives (real and imaginary parts)
        D_x_phi_real = d_phi_mag_dx * np.cos(field_phase) - field_mag * (d_phi_phase_dx + self.g_coupling * self.A_x) * np.sin(field_phase)
        D_x_phi_imag = d_phi_mag_dx * np.sin(field_phase) + field_mag * (d_phi_phase_dx + self.g_coupling * self.A_x) * np.cos(field_phase)
        
        D_y_phi_real = d_phi_mag_dy * np.cos(field_phase) - field_mag * (d_phi_phase_dy + self.g_coupling * self.A_y) * np.sin(field_phase)
        D_y_phi_imag = d_phi_mag_dy * np.sin(field_phase) + field_mag * (d_phi_phase_dy + self.g_coupling * self.A_y) * np.cos(field_phase)
        
        return D_x_phi_real, D_x_phi_imag, D_y_phi_real, D_y_phi_imag
    
    def compute_lagrangian_density(self):
        """Compute U(1) gauge theory Lagrangian density"""
        # Covariant kinetic term |D_μ φ|²
        Dx_real, Dx_imag, Dy_real, Dy_imag = self.covariant_derivative(self.phi_mag, self.phi_phase)
        kinetic_term = Dx_real**2 + Dx_imag**2 + Dy_real**2 + Dy_imag**2
        
        # Potential term V(|φ|²) = λ(|φ|² - v²)²
        v_squared = 1.0  # Vacuum expectation value
        potential_term = self.lambda_potential * (self.phi_mag**2 - v_squared)**2
        
        # Maxwell term -¼ F_μν F^μν
        maxwell_term = -0.25 * (self.F_xy**2 + self.F_tx**2 + self.F_ty**2)
        
        # Total Lagrangian density
        lagrangian_density = kinetic_term - potential_term + maxwell_term
        
        return lagrangian_density, kinetic_term, potential_term, maxwell_term
    
    def strand_selection_measurement(self):
        """Measurement event: select coherent strand cluster (gauge fixing)"""
        # Random measurement location
        meas_x = np.random.uniform(0.2*self.grid_size, 0.8*self.grid_size)
        meas_y = np.random.uniform(0.2*self.grid_size, 0.8*self.grid_size)
        
        # Find strands near measurement point
        selection_radius = 15
        coherences = []
        
        for i, strand in enumerate(self.rope_strands):
            # Distance to measurement point
            strand_2d = strand[:, :2]
            distances = np.sqrt((strand_2d[:, 0] - meas_x)**2 + (strand_2d[:, 1] - meas_y)**2)
            min_dist = np.min(distances)
            
            if min_dist < selection_radius:
                coherence = np.exp(-min_dist**2 / (selection_radius/2)**2)
                coherences.append(coherence)
                self.strand_coherences[i] = coherence
            else:
                coherences.append(0.1)
                self.strand_coherences[i] = 0.1
        
        # Select dominant strand
        self.selected_strand_idx = np.argmax(coherences)
        
        # Normalize coherences
        total_coherence = sum(self.strand_coherences)
        if total_coherence > 0:
            self.strand_coherences = [c/total_coherence for c in self.strand_coherences]
    
    def evolve_scout_dynamics(self):
        """Evolve scout as charged particle in gauge field"""
        sx, sy = int(self.scout_pos[0]), int(self.scout_pos[1])
        sx = np.clip(sx, 0, self.grid_size-1)
        sy = np.clip(sy, 0, self.grid_size-1)
        
        # Lorentz force: F = q(E + v × B)
        # Electric field E = -∇φ - ∂A/∂t (simplified)
        Ex = -self.F_tx[sy, sx]
        Ey = -self.F_ty[sy, sx]
        
        # Magnetic field B = ∇ × A
        Bz = self.F_xy[sy, sx]
        
        # Lorentz force
        force_x = self.scout_charge * (Ex + self.scout_vel[1] * Bz)
        force_y = self.scout_charge * (Ey - self.scout_vel[0] * Bz)
        
        # Update scout dynamics
        self.scout_vel += np.array([force_x, force_y]) * self.dt / self.scout_mass
        self.scout_vel *= 0.98  # Damping
        self.scout_pos += self.scout_vel * self.dt
        
        # Boundary conditions
        self.scout_pos[0] = np.clip(self.scout_pos[0], 0, self.grid_size-1)
        self.scout_pos[1] = np.clip(self.scout_pos[1], 0, self.grid_size-1)
    
    def compute_wilson_loop(self):
        """Compute Wilson loop for gauge invariant observable"""
        # Simple rectangular Wilson loop
        loop_size = 10
        center_x, center_y = self.grid_size//2, self.grid_size//2
        
        # Path integral around loop
        wilson_phase = 0.0
        
        # Bottom edge
        for x in range(center_x - loop_size//2, center_x + loop_size//2):
            if 0 <= x < self.grid_size and 0 <= center_y - loop_size//2 < self.grid_size:
                wilson_phase += self.A_x[center_y - loop_size//2, x] * self.dx
        
        # Right edge  
        for y in range(center_y - loop_size//2, center_y + loop_size//2):
            if 0 <= center_x + loop_size//2 < self.grid_size and 0 <= y < self.grid_size:
                wilson_phase += self.A_y[y, center_x + loop_size//2] * self.dx
        
        # Top edge (reverse)
        for x in range(center_x + loop_size//2, center_x - loop_size//2, -1):
            if 0 <= x < self.grid_size and 0 <= center_y + loop_size//2 < self.grid_size:
                wilson_phase -= self.A_x[center_y + loop_size//2, x] * self.dx
        
        # Left edge (reverse)
        for y in range(center_y + loop_size//2, center_y - loop_size//2, -1):
            if 0 <= center_x - loop_size//2 < self.grid_size and 0 <= y < self.grid_size:
                wilson_phase -= self.A_y[y, center_x - loop_size//2] * self.dx
        
        return np.exp(1j * self.g_coupling * wilson_phase)
    
    def step(self):
        """Evolution step"""
        # Periodic strand selection (measurement events)
        if int(self.time / self.dt) % 200 == 100:  # Every 200 steps
            self.strand_selection_measurement()
        
        # Update gauge potential from rope geometry
        self.compute_gauge_potential_from_rope()
        
        # Compute field strength
        self.compute_field_strength()
        
        # Evolve complex scalar field (simplified)
        # This should be the full Klein-Gordon equation with gauge coupling
        # For now, simplified evolution
        laplacian_mag = self.laplacian_2d(self.phi_mag)
        laplacian_phase = self.laplacian_2d(self.phi_phase)
        
        # Equations of motion (simplified)
        self.phi_mag += self.dt * (0.5 * laplacian_mag - self.lambda_potential * self.phi_mag * (self.phi_mag**2 - 1))
        self.phi_phase += self.dt * (0.3 * laplacian_phase)
        
        # Evolve scout
        self.evolve_scout_dynamics()
        
        # Compute observables
        lagrangian_density, _, _, _ = self.compute_lagrangian_density()
        total_lagrangian = np.sum(lagrangian_density)
        self.lagrangian_density_history.append(total_lagrangian)
        
        # Wilson loop
        wilson_loop = self.compute_wilson_loop()
        self.wilson_loop_history.append(np.abs(wilson_loop))
        
        # Gauge invariant quantity
        gauge_invariant = np.sum(self.phi_mag**2)
        self.gauge_invariant_history.append(gauge_invariant)
        
        self.time += self.dt
    
    def laplacian_2d(self, field):
        """2D Laplacian with periodic boundary conditions"""
        lap_x = (np.roll(field, -1, axis=1) - 2*field + np.roll(field, 1, axis=1)) / self.dx**2
        lap_y = (np.roll(field, -1, axis=0) - 2*field + np.roll(field, 1, axis=0)) / self.dx**2
        return lap_x + lap_y

def create_u1_gauge_visualizer():
    """Create comprehensive U(1) gauge theory visualizer"""
    system = U1GaugeRopeField(grid_size=80, n_strands=6)
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('U(1) Gauge Theory from Fractal Rope + COM Instanton', fontsize=16, fontweight='bold')
    
    # Complex scalar field
    ax1 = plt.subplot(3, 4, 1)
    phi_mag_img = ax1.imshow(system.phi_mag, cmap='plasma', origin='lower')
    ax1.set_title('Scalar Field |φ|')
    plt.colorbar(phi_mag_img, ax=ax1, shrink=0.6)
    
    ax2 = plt.subplot(3, 4, 2)
    phi_phase_img = ax2.imshow(system.phi_phase, cmap='hsv', origin='lower')
    ax2.set_title('Scalar Field Phase θ')
    plt.colorbar(phi_phase_img, ax=ax2, shrink=0.6)
    
    # Gauge potential
    ax3 = plt.subplot(3, 4, 3)
    ax3.set_title('Gauge Potential A_μ')
    ax3.set_xlim(0, system.grid_size)
    ax3.set_ylim(0, system.grid_size)
    
    # Field strength
    ax4 = plt.subplot(3, 4, 4)
    field_strength_img = ax4.imshow(system.F_xy, cmap='RdBu_r', origin='lower')
    ax4.set_title('Field Strength F_xy')
    plt.colorbar(field_strength_img, ax=ax4, shrink=0.6)
    
    # Lagrangian density
    ax5 = plt.subplot(3, 4, 5)
    ax5.set_title('Lagrangian Density')
    lagrangian_line, = ax5.plot([], [], 'b-', linewidth=2)
    ax5.grid(True)
    
    # Wilson loop
    ax6 = plt.subplot(3, 4, 6)
    ax6.set_title('Wilson Loop |W|')
    wilson_line, = ax6.plot([], [], 'r-', linewidth=2)
    ax6.grid(True)
    
    # Gauge invariant
    ax7 = plt.subplot(3, 4, 7)
    ax7.set_title('Gauge Invariant ∫|φ|²')
    gauge_line, = ax7.plot([], [], 'g-', linewidth=2)
    ax7.grid(True)
    
    # Rope strands 3D
    ax8 = plt.subplot(3, 4, 8, projection='3d')
    ax8.set_title('Fractal Rope Strands')
    
    # Scout trajectory
    ax9 = plt.subplot(3, 4, 9)
    ax9.set_title('Scout Trajectory')
    scout_line, = ax9.plot([], [], 'ro-', markersize=4, linewidth=1)
    ax9.set_xlim(0, system.grid_size)
    ax9.set_ylim(0, system.grid_size)
    
    # Strand coherences
    ax10 = plt.subplot(3, 4, 10)
    ax10.set_title('Strand Coherences')
    coherence_bars = ax10.bar(range(system.n_strands), system.strand_coherences)
    ax10.set_ylim(0, 1)
    
    # Theory summary
    ax11 = plt.subplot(3, 4, (11, 12))
    ax11.axis('off')
    theory_text = ax11.text(0.05, 0.95, '', transform=ax11.transAxes,
                           fontfamily='monospace', verticalalignment='top')
    
    # Animation data
    scout_trajectory = []
    max_history = 150
    
    def animate(frame):
        # Evolve system
        for _ in range(2):
            system.step()
        
        # Update scalar field displays
        phi_mag_img.set_array(system.phi_mag)
        phi_mag_img.set_clim(vmin=0, vmax=np.max(system.phi_mag))
        
        phi_phase_img.set_array(system.phi_phase)
        
        # Update gauge potential visualization
        ax3.clear()
        ax3.set_title('Gauge Potential A_μ')
        
        # Show vector field
        x_coords = np.arange(0, system.grid_size, 4)
        y_coords = np.arange(0, system.grid_size, 4)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Subsample gauge potential for arrows
        Ax_sub = system.A_x[::4, ::4]
        Ay_sub = system.A_y[::4, ::4]
        
        ax3.quiver(X, Y, Ax_sub, Ay_sub, scale=5, color='blue', alpha=0.7)
        ax3.set_xlim(0, system.grid_size)
        ax3.set_ylim(0, system.grid_size)
        
        # Update field strength
        field_strength_img.set_array(system.F_xy)
        field_strength_img.set_clim(vmin=np.min(system.F_xy), vmax=np.max(system.F_xy))
        
        # Update time series
        if len(system.lagrangian_density_history) > 1:
            recent_lag = system.lagrangian_density_history[-max_history:]
            lagrangian_line.set_data(range(len(recent_lag)), recent_lag)
            ax5.set_xlim(0, len(recent_lag))
            ax5.set_ylim(min(recent_lag), max(recent_lag))
        
        if len(system.wilson_loop_history) > 1:
            recent_wilson = system.wilson_loop_history[-max_history:]
            wilson_line.set_data(range(len(recent_wilson)), recent_wilson)
            ax6.set_xlim(0, len(recent_wilson))
            ax6.set_ylim(0, max(recent_wilson) * 1.1)
        
        if len(system.gauge_invariant_history) > 1:
            recent_gauge = system.gauge_invariant_history[-max_history:]
            gauge_line.set_data(range(len(recent_gauge)), recent_gauge)
            ax7.set_xlim(0, len(recent_gauge))
            ax7.set_ylim(min(recent_gauge) * 0.9, max(recent_gauge) * 1.1)
        
        # Update 3D rope strands
        ax8.clear()
        ax8.set_title('Fractal Rope Strands')
        
        for i, strand in enumerate(system.rope_strands):
            coherence = system.strand_coherences[i]
            color = 'red' if i == system.selected_strand_idx else 'blue'
            alpha = 0.3 + 0.7 * coherence
            ax8.plot(strand[:, 0], strand[:, 1], strand[:, 2], 
                    color=color, alpha=alpha, linewidth=1+2*coherence)
        
        # Update scout trajectory
        scout_trajectory.append(system.scout_pos.copy())
        if len(scout_trajectory) > 50:
            scout_trajectory.pop(0)
        
        if len(scout_trajectory) > 1:
            traj_x = [pos[0] for pos in scout_trajectory]
            traj_y = [pos[1] for pos in scout_trajectory]
            scout_line.set_data(traj_x, traj_y)
        
        # Update coherence bars
        for i, bar in enumerate(coherence_bars):
            bar.set_height(system.strand_coherences[i])
            bar.set_color('red' if i == system.selected_strand_idx else 'blue')
        
        # Update theory text
        lagrangian_density, kinetic, potential, maxwell = system.compute_lagrangian_density()
        total_action = np.sum(lagrangian_density)
        
        theory_str = f"""U(1) GAUGE THEORY ANALYSIS

LAGRANGIAN: L = |D_μφ|² - V(|φ|²) - ¼F_μνF^μν

FIELD CONFIGURATION:
• Scalar field magnitude: |φ| ∈ [0, {np.max(system.phi_mag):.2f}]
• Phase winding: θ ∈ [-π, π]
• Gauge coupling: g = {system.g_coupling}

GAUGE POTENTIAL:
• A_x range: [{np.min(system.A_x):.3f}, {np.max(system.A_x):.3f}]
• A_y range: [{np.min(system.A_y):.3f}, {np.max(system.A_y):.3f}]
• Field strength: F_xy ∈ [{np.min(system.F_xy):.3f}, {np.max(system.F_xy):.3f}]

DYNAMICS:
• Total action: S = {total_action:.2f}
• Kinetic energy: {np.sum(kinetic):.2f}
• Potential energy: {np.sum(potential):.2f}
• Maxwell energy: {np.sum(maxwell):.2f}

OBSERVABLES:
• Wilson loop: |W| = {system.wilson_loop_history[-1] if system.wilson_loop_history else 0:.3f}
• Gauge invariant: ∫|φ|² = {system.gauge_invariant_history[-1] if system.gauge_invariant_history else 0:.2f}
• Selected strand: {system.selected_strand_idx + 1}/{system.n_strands}

EMERGENCE MECHANISM:
• Rope geometry → Berry connection → A_μ
• Strand selection → Gauge fixing
• Scout charge → Minimal coupling
• Complex instanton → Higgs-like field
"""
        
        theory_text.set_text(theory_str)
        
        return [phi_mag_img, phi_phase_img, field_strength_img, lagrangian_line, 
                wilson_line, gauge_line, scout_line, theory_text]
    
    ani = FuncAnimation(fig, animate, frames=1000, interval=150, blit=False, repeat=True)
    plt.tight_layout()
    
    return fig, ani, system

if __name__ == "__main__":
    print("🌌 U(1) Gauge Theory from Fractal Rope + COM-Instanton")
    print("📐 Geometric gauge field emergence")
    print("🎯 Wilson loops and gauge invariants")
    print("⚡ Complex scalar field dynamics")
    print("🧬 Strand selection = Gauge fixing")
    
    fig, ani, system = create_u1_gauge_visualizer()
    plt.show()