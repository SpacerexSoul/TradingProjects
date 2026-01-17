import matplotlib.pyplot as plt
import numpy as np


def plot_evolution(all_generations, output_path="evolution.png"):
    """Plot all individuals across generations - shows evolution visually."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    num_gens = len(all_generations)
    colors = plt.cm.viridis(np.linspace(0, 1, num_gens))
    
    for gen_idx, gen_data in enumerate(all_generations):
        fitness = gen_data['fitness']
        x = [gen_idx + 1] * len(fitness)
        
        # Scatter all individuals
        ax.scatter(x, fitness, c=[colors[gen_idx]], alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    # Highlight best per generation
    best_per_gen = [g['best'] for g in all_generations]
    ax.plot(range(1, num_gens + 1), best_per_gen, 'r-', linewidth=2, label='Best per Gen', zorder=5)
    ax.scatter(range(1, num_gens + 1), best_per_gen, c='red', s=100, zorder=6, edgecolors='black')
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('ROI (%)', fontsize=12)
    ax.set_title('Genetic Algorithm Evolution: Population ROI per Generation', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Color bar to show generation progression
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(1, num_gens))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Generation')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")
    plt.close()
