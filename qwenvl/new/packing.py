import json
import os
import time
import binpacking
import datasets
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoProcessor
"""
This module implements a fast version of the Best Fit Decreasing algorithm for bin packing.
It uses a segment tree data structure to efficiently find the best bin for each item.
The algorithm is optimized for performance using Numba's just-in-time compilation.

Author: Qingyang Wu
"""

import numpy as np
import tqdm
import math
from numba import njit

DEBUG = os.environ.get('DEBUG', '0') == '1'

@njit(cache=True)
def get_height(num_bins: int) -> int:
  if num_bins == 0:
    return 1
  return math.floor(math.log2(num_bins) + 1) + 1


@njit(cache=True)
def build_tree(bins_remaining_capacity: np.ndarray, segment_tree: np.ndarray, num_bins: int):
  # Calculate the total height of the segment tree
  total_height = get_height(num_bins)

  # Start filling the segment tree from the leaves
  for i in range(num_bins):
    leaf_index = (2 ** (total_height - 1) - 1) + i
    segment_tree[leaf_index] = bins_remaining_capacity[i]

  # Now fill the internal nodes of the segment tree
  # Start from the second last level and go upwards
  for h in range(total_height - 2, -1, -1):
    level_start = 2 ** h - 1
    level_end = 2 ** (h + 1) - 1

    for i in range(level_start, level_end):
      left = 2 * i + 1
      right = 2 * i + 2
      segment_tree[i] = max(segment_tree[left], segment_tree[right])


@njit(cache=True)
def update_tree(bins_remaining_capacity: np.ndarray, segment_tree: np.ndarray, num_bins: int, bin_index: int, value: int):
  node = 0
  height = 0
  start = 0
  # end = len(bins_remaining_capacity) - 1
  end = 2 ** (get_height(num_bins - 1) - 1) - 1

  while True:
    # neeed to count one less than the number of bins
    if height == get_height(num_bins - 1) - 1:
      bins_remaining_capacity[bin_index] = segment_tree[node] = value
      break
    else:
      left = 2 * node + 1
      right = 2 * node + 2
      mid_bin_index = (start + end) // 2
      if bin_index <= mid_bin_index:
        node = left
        height += 1
        end = mid_bin_index
      else:
        node = right
        height += 1
        start = mid_bin_index + 1

  # Update parent nodes
  while node > 0:
    node = (node - 1) // 2
    left = 2 * node + 1
    right = 2 * node + 2
    segment_tree[node] = max(segment_tree[left], segment_tree[right])


@njit(cache=True)
def search(segment_tree: np.ndarray, value: int, num_bins: int):
  height = 0
  node = 0
  while height < get_height(num_bins) - 1:
    left = 2 * node + 1
    right = 2 * node + 2
    if segment_tree[left] >= value:
      node = left
    else:
      node = right
    height += 1
  return segment_tree_to_bin_index(node, num_bins)


@njit(cache=True)
def segment_tree_to_bin_index(node: int, num_bins: int):
  return node - (2 ** (get_height(num_bins) - 1) - 1)

# @njit(cache=True)


def insert(num_items: int, num_bins: int, bins_remaining_capacity: np.ndarray, segment_tree: np.ndarray, item: int):
  if segment_tree[0] < item:
    build_tree(bins_remaining_capacity, segment_tree, num_bins=num_bins)

  bin_index = search(segment_tree, value=item, num_bins=num_bins)
  if bin_index >= num_bins:
    bin_index = num_bins
    num_bins += 1

  bins_remaining_capacity[bin_index] -= item
  # item_to_bin.append(bin_index)
  update_tree(bins_remaining_capacity, segment_tree, num_bins,
              bin_index, bins_remaining_capacity[bin_index])

  num_items += 1

  return bin_index, num_items, num_bins


def fast_best_fit_decreasing(items, bin_capacity, batch_size=50000, progress_bar=True):
  bin_capacity = bin_capacity
  
  def _fast_best_fit_decreasing(items, bin_capacity):
    num_items = 0
    num_bins = 0
    bin_indices = []

    bins_remaining_capacity = np.array([bin_capacity] * len(items))
    segment_tree = np.array([bin_capacity] * (4 * len(items) + 5))

    indexed_items = sorted(enumerate(items), key=lambda x: x[1], reverse=True)

    for _, item in tqdm.tqdm(indexed_items, desc="Packing items", disable=not progress_bar):
      bin_index, num_items, num_bins = insert(
          num_items, num_bins, bins_remaining_capacity, segment_tree, item)
      bin_indices.append(bin_index)
      
    # W build the bins to items mapping
    bins_to_items = [[] for _ in range(num_bins)]
    for item_index, bin_index in enumerate(bin_indices):
      bins_to_items[bin_index].append(indexed_items[item_index][0])

    return bins_to_items
  
  bins = list()
  for slice_start in range(0, len(items), batch_size):
    slice_end = min(slice_start + batch_size, len(items))
    slice_items = items[slice_start:slice_end]
    slice_bins = _fast_best_fit_decreasing(slice_items, bin_capacity)
    slice_bins = [[slice_start + i for i in slice_bin] for slice_bin in slice_bins]
    bins.extend(slice_bins)
  return bins