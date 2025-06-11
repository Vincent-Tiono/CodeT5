import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

def uniform_sphere_points(num_points, num_dim):
    # Sample points from a uniform distribution on the surface of a unit hypersphere
    # https://stats.stackexchange.com/questions/7977/how-to-generate-uniformly-distributed-points-on-the-surface-of-the-3-d-unit-sphe
    points = torch.randn(num_points, num_dim)
    # Normalize points to a unit hypersphere
    points = points / torch.norm(points, dim=1, keepdim=True)
    return points

if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    points = uniform_sphere_points(100000, 256).to(device)
    positions = torch.zeros(100000, 100000)
    for i, p in tqdm(enumerate(points)):
        sim = torch.sum(points * p, dim=1)
        indices = torch.argsort(sim, descending=True).cpu()
        positions[i, indices] = torch.arange(100000).float()
    
    average_pos = torch.mean(positions, dim=0).numpy()
    sns.distplot(average_pos)
    plt.xlabel("Average Position")
    plt.ylabel("Density")
    plt.show()
