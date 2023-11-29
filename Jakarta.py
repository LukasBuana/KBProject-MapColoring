import cv2
import numpy as np
import networkx as nx
from PIL import Image
from sklearn.cluster import KMeans
from skimage.measure import label, regionprops

img = cv2.imread('Prov._DKI_Jakarta.jpg')
desired_size = 2000
img = cv2.resize(img, (desired_size, desired_size))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

target_color = np.array([212, 212, 212], dtype=np.uint8)
color_tolerance = 40
mask = cv2.inRange(img_rgb, target_color - color_tolerance, target_color + color_tolerance)

labeled_mask = label(mask)

kmeans = KMeans(n_clusters=6, n_init=10)
kmeans.fit(img_rgb.reshape((-1, 3)))
labels = kmeans.labels_

G = nx.Graph()
regions = regionprops(labeled_mask)
for region in regions:
    label_value = region.label
    G.add_node(label_value)

for region1 in regions:
    label_value1 = region1.label
    bbox1 = region1.bbox
    for region2 in regions:
        label_value2 = region2.label
        bbox2 = region2.bbox
        if label_value1 != label_value2 and (
            bbox1[0] <= bbox2[2] and bbox1[2] >= bbox2[0] and bbox1[1] <= bbox2[3] and bbox1[3] >= bbox2[1]
        ):
            G.add_edge(label_value1, label_value2)

coloring = nx.coloring.greedy_color(G, strategy="largest_first")

img_array = np.copy(img)

used_colors = {0}

predefined_colors = [
    np.array([255, 0, 0], dtype=np.uint8),   # Red
    np.array([0, 255, 0], dtype=np.uint8),   # Green
    np.array([0, 0, 255], dtype=np.uint8),   # Blue
    np.array([255, 255, 0], dtype=np.uint8), # Yellow
    np.array([255, 0, 255], dtype=np.uint8), # Magenta
    np.array([0, 255, 255], dtype=np.uint8), # Cyan
]

for region in regions:
    label_value = region.label
    color_index = coloring[label_value]

    if np.linalg.norm(np.mean(img_rgb[labeled_mask == label_value], axis=0) - [235, 235, 235]) < color_tolerance:
        continue

    if color_index != -1:
        target_region_indices = np.where(labeled_mask == label_value)
        target_region_mask = np.zeros_like(labeled_mask, dtype=np.uint8)
        target_region_mask[target_region_indices] = 1

        # Ensure different colors for adjacent regions
        neighbors = list(G.neighbors(label_value))
        neighbor_colors = set(coloring[neighbor] for neighbor in neighbors)
        available_colors = set(range(6)) - neighbor_colors

        if not available_colors:
            new_color = max(used_colors) + 1
        else:
            new_color = min(available_colors)

        used_colors.add(new_color)
        coloring[label_value] = new_color

        color_index = coloring[label_value]
        color_index %= len(predefined_colors)

        color = predefined_colors[color_index]
        img_array[target_region_mask == 1] = color

output_image = Image.fromarray(img_array)

output_image.save('jakarta_colored.png')

output_image.show()