import math
import matplotlib.pyplot as plt

def find_obstacles(points):
    obstacles = []
    for x, y, z in points:
        distance = math.sqrt(x**2 + y**2)
        if distance < 10:
            obstacles.append((x, y, z))
    return obstacles

def plot_points(all_points, obstacle_points):
    x_all = [p[0] for p in all_points]
    y_all = [p[1] for p in all_points]
    x_obs = [p[0] for p in obstacle_points]
    y_obs = [p[1] for p in obstacle_points]

    plt.figure(figsize=(8, 8))
    plt.scatter(x_all, y_all, c='gray', label='All Points')
    plt.scatter(x_obs, y_obs, c='red', label='Obstacles (<10m)', marker='x')
    plt.scatter(0, 0, c='blue', label='Robot Origin')
    plt.gca().add_patch(plt.Circle((0, 0), 10, color='blue', fill=False, linestyle='--', label='10m Range'))
    plt.title('Obstacle Detection from LIDAR Data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.savefig('lidar_visual.png')
    plt.show()

if __name__ == "__main__":
    # Sample data
    lidar_points = [
        (10, 0, 0), (1, 1, 0), (0, 10, 0), 
        (6, 8, 0), (8, 6, 0), (-8, 6, 0), 
        (1, -5, 0), (11, 11, 0), (3, 4, 0), (-7, -7, 0)
    ]

    obstacles = find_obstacles(lidar_points)
    print("Detected Obstacles:", obstacles)
    plot_points(lidar_points, obstacles)
