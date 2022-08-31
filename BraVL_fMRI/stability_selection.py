import numpy as np
from itertools import combinations

def stability_selection(data, n=None):
    """Return the indices of the n voxels with best stability

    Given repeated fMRI measurements on a set of stimuli, return the indices of
    the voxels that demonstrate the best stability across the repetitions. This
    stability is quantified for each voxel, as the mean Pearson correlation
    coefficient across all pairwise combinations of the repetions.

    Parameters
    ----------
    data : 3D array (n_repetitions, n_items, n_voxels)
        The fMRI images
    n : int | None
        If specified, the indices of the top N most stable vertices are
        returned. Otherwise the indices of all vertices are returned.

    Returns
    -------
    top_indices : 1D array (n_voxels)
        The indices of the vertices, ordered by stability in decreasing order
        (the first index corresponds to the vertex with the highest stability).
        If the n parameter is specified, as most N indices are returned.
    """
    n_repetitions, n_items, n_voxels = data.shape

    if n is None:
        n = n_voxels
    elif n > n_voxels:
        raise ValueError('n must be a number between 0 and ' + n_voxels)

    # Drop all voxels don't contain NaN's for any items
    non_nan_mask = ~np.any(np.any(np.isnan(data), axis=1), axis=0)
    non_nan_indices = np.flatnonzero(non_nan_mask)
    data_trimmed = data[:, :, non_nan_mask]

    data_means = data_trimmed.mean(axis=1)
    data_stds = data_trimmed.std(axis=1)

    # Loop over all pairwise combinations and compute correlations
    stability_scores = []
    for x, y in combinations(range(n_repetitions), 2):
        x1 = (data_trimmed[x] - data_means[x]) / data_stds[x]
        y1 = (data_trimmed[y] - data_means[y]) / data_stds[y]
        stability_scores.append(np.sum(x1 * y1, axis=0) / n_items)

    # Compute the N best voxels
    best_voxels = np.mean(stability_scores, axis=0).argsort()[-n:]

    # Return the (original) indices of the best voxels in decreasing order
    return non_nan_indices[best_voxels][::-1]
