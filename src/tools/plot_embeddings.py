""" General tools shared across the project. """
import os
import copy

from matplotlib.colors import ListedColormap
from collections import defaultdict
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import umap
from torch import nn

def umap_embeddings(p_outdir, p_embds, p_labels, title='Embeddings Visualization',
                    n_neighbors=30, min_dist=0.0, n_components=2, random_state=None,
                    spread=1.0):
    """
    Visualize the embeddings using UMAP
    :param p_embds: The embeddings
    :param p_labels: The labels
    :param title: The title of the plot
    :param n_neighbors: The number of neighbors to consider
    :param min_dist: The minimum distance between points
    :param n_components: The number of components (the number of dimensions)
    :param random_state: The random state for reproducibility
    """
    print(f"Rendering UMAP plot with {n_components} components,"
          f"{n_neighbors} neighbors, and {min_dist} min_dist")

    umap_obj = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
                         random_state=random_state, spread=spread)
    p_embds = umap_obj.fit_transform(p_embds)

    # For labels that might contain multiple parts (e.g., "M-Verb3"), we'll use the first part
    # as the primary grouping for color families, and the full label for the specific color

    # Generate a color map for unique labels
    unique_labels = sorted(set(p_labels))

    # Extract primary categories from labels (e.g., extract "M" from "M-Verb3")
    # This handles both single category labels and multi-part labels
    primary_categories = sorted(set(label.split('-')[0] if '-' in label else label 
                               for label in unique_labels))

    # Create a color map for primary categories
    custom_colors = ['red', 'green', 'blue', 'purple', 'cyan', 'orange', 'brown', 'magenta', 
                     'yellow', 'teal', 'pink', 'olive', 'navy', 'coral']
    # Extend colors if we have more categories than colors
    while len(custom_colors) < len(primary_categories):
        custom_colors.extend(custom_colors)

    color_map = ListedColormap(custom_colors[:len(primary_categories)])
    base_colors = {cat: color_map(i % len(primary_categories)) 
                  for i, cat in enumerate(primary_categories)}

    # Group labels by their primary category
    label_groups = defaultdict(list)
    for label in unique_labels:
        primary = label.split('-')[0] if '-' in label else label
        label_groups[primary].append(label)

    # Generate colors for each label
    label_colors = {}
    for primary, group_labels in label_groups.items():
        sorted_group = sorted(group_labels)
        num_in_group = len(sorted_group)

        for i, group_label in enumerate(sorted_group):
            # Calculate the color blend between the base color and white
            base_color = base_colors[primary]
            blend_factor = (i + 1) / (num_in_group + 1)  # +1 to avoid pure white
            # Blend towards white
            blended_color = [1 - (1 - bc) * blend_factor for bc in base_color[:3]]
            label_colors[group_label] = blended_color

    # Apply colors to the points
    colors = [label_colors[label] for label in p_labels]

    plt.figure(figsize=(10, 10))
    plt.scatter(p_embds[:, 0], p_embds[:, 1], c=colors)

    # Create legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=label_colors[label], markersize=10)
                         for label in unique_labels]
    plt.legend(handles, unique_labels, loc='best', ncol=2 if len(unique_labels) > 10 else 1, prop={'size': 20})
    plt.title(title, fontsize=24)

    plt.tight_layout()
    plt.savefig(os.path.join(p_outdir, f"{title}.png"))
    plt.close()

def tsne_embeddings(p_outdir, p_embds, p_labels, title='Embeddings Visualization',
                    n_components=2, perplexity=50):
    """
    Visualize the embeddings using t-SNE
    :param p_embds: The embeddings
    :param p_labels: The labels
    :param title: The title of the plot
    :param n_components: The number of components (the number of dimensions)
    :param perplexity: The perplexity of the t-SNE (the number of nearest neighbors to consider)
    """
    print(f"Rendering t-SNE plot with {n_components} components and {perplexity} perplexity")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    p_embds = tsne.fit_transform(p_embds)

    # Extract unique first parts and create color map (same as UMAP)
    first_parts = sorted(set(label.split('-')[0] for label in p_labels))
    custom_colors = ['red', 'green', 'blue', 'purple', 'cyan', 'orange', 'brown']
    color_map = ListedColormap(custom_colors)
    base_colors = {first_part: color_map(i / len(first_parts)) for i, first_part in enumerate(first_parts)}

    # Dictionary to hold second parts for each first part
    second_parts_dict = defaultdict(list)
    for label in p_labels:
        first_part, second_part = label.split('-')
        second_parts_dict[first_part].append(second_part)

    # Generate colors for each label
    label_colors = {}
    for first_part, second_parts in second_parts_dict.items():
        unique_second_parts = sorted(set(second_parts))
        num_unique_second_parts = len(unique_second_parts)

        for i, second_part in enumerate(unique_second_parts):
            # Calculate the color blend between the base color and white
            base_color = base_colors[first_part]
            blend_factor = (i + 1) / num_unique_second_parts
            # blend towards white
            blended_color = [1 - (1 - bc) * blend_factor for bc in base_color[:3]]

            for label in p_labels:
                if label == f"{first_part}-{second_part}":
                    label_colors[label] = blended_color

    # Apply colors to the labels
    colors = [label_colors[label] for label in p_labels]

    plt.figure(figsize=(10, 10))
    plt.scatter(p_embds[:, 0], p_embds[:, 1], c=colors)

    # Create legend
    unq_labels = sorted(list(set(p_labels)))
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=label_colors[label], markersize=10)
                          for label in unq_labels]
    plt.legend(handles, unq_labels)
    plt.title(title)
    plt.savefig(os.path.join(p_outdir, f"{title}.png"))
    plt.close()


# Define early stopping criteria
class EarlyStopping:
    """
    Class to implement early stopping criteria during model training
    """
    def __init__(self, patience: int = 100, min_delta: float = 0.0):
        """
        Initialize the early stopping criteria
        :param patience: number of epochs to wait before stopping
        :param min_delta: minimum change in loss to be considered as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.epochs_no_improve = 0
        self.early_stop = False
        self.best_model = None
        self.min_loss = None

    def __call__(self, val_loss: float, model: nn.Module):
        """ 
        Check if the model should stop training 
        :param val_loss: validation loss
        :param model: model to be saved
        """
        if self.min_loss is None:
            self.min_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.min_loss - self.min_delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
        else:
            self.min_loss = val_loss
            self.save_checkpoint(model)
            self.epochs_no_improve = 0
            print(f"New min loss: {val_loss}")

    def save_checkpoint(self, model: nn.Module):
        """
        Save the model if there is an improvement
        :param model: model to be saved
        """
        self.best_model = copy.deepcopy(model.state_dict())
