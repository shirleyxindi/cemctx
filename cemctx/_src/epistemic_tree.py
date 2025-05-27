# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A data structure used to hold / inspect search data for a batch of inputs."""

from __future__ import annotations
from typing import Any, ClassVar, Generic, TypeVar

import chex
import jax
import jax.numpy as jnp

from cemctx._src.tree import SearchSummary


T = TypeVar("T")


@chex.dataclass(frozen=True)
class EpistemicTree(Generic[T]):
  """State of a search tree.

  The `Tree` dataclass is used to hold and inspect search data for a batch of
  inputs. In the fields below `B` denotes the batch dimension, `N` represents
  the number of nodes in the tree, and `num_actions` is the number of discrete
  actions.

  node_visits: `[B, N]` the visit counts for each node.
  raw_values: `[B, N]` the raw value for each node.
  node_values: `[B, N]` the cumulative search value for each node.
  parents: `[B, N]` the node index for the parents for each node.
  action_from_parent: `[B, N]` action to take from the parent to reach each
    node.
  children_index: `[B, N, num_actions]` the node index of the children for each
    action.
  children_prior_logits: `[B, N, num_actions]` the action prior logits of each
    node.
  children_visits: `[B, N, num_actions]` the visit counts for children for
    each action.
  children_rewards: `[B, N, num_actions]` the immediate reward for each action.
  children_discounts: `[B, N, num_actions]` the discount between the
    `children_rewards` and the `children_values`.
  children_values: `[B, N, num_actions]` the value of the next node after the
    action.
  embeddings: `[B, N, ...]` the state embeddings of each node.
  root_invalid_actions: `[B, num_actions]` a mask with invalid actions at the
    root. In the mask, invalid actions have ones, and valid actions have zeros.
  extra_data: `[B, ...]` extra data passed to the search.
  """
  node_visits: chex.Array  # [B, N]
  raw_values: chex.Array  # [B, N]
  node_values: chex.Array  # [B, N]
  parents: chex.Array  # [B, N]
  action_from_parent: chex.Array  # [B, N]
  children_index: chex.Array  # [B, N, num_actions]
  children_prior_logits: chex.Array  # [B, N, num_actions]
  children_visits: chex.Array  # [B, N, num_actions]
  children_rewards: chex.Array  # [B, N, num_actions]
  children_discounts: chex.Array  # [B, N, num_actions]
  children_values: chex.Array  # [B, N, num_actions]
  embeddings: Any  # [B, N, ...]
  root_invalid_actions: chex.Array  # [B, num_actions]
  extra_data: T  # [B, ...]

  # For EMCTS:
  raw_values_epistemic_variance: chex.Array  # [B, N]
  node_values_epistemic_std: chex.Array  # [B, N]
  children_rewards_epistemic_variance: chex.Array  # [B, N, num_actions]
  children_values_epistemic_std: chex.Array  # [B, N, num_actions]
  beta_v: chex.Array   # [B]

  # For CEMCTS:
  raw_cost_values: chex.Array  # [B, N]
  node_cost_values: chex.Array  # [B, N]
  children_costs: chex.Array  # [B, N, num_actions]
  children_cost_values: chex.Array  # [B, N, num_actions]
  raw_cost_values_epistemic_variance: chex.Array  # [B, N]
  node_cost_values_epistemic_std: chex.Array  # [B, N]
  children_costs_epistemic_variance: chex.Array  # [B, N, num_actions]
  children_cost_values_epistemic_std: chex.Array  # [B, N, num_actions]
  beta_c: chex.Array   # [B]
  cost_threshold: chex.Array  # [B]

  # The following attributes are class variables (and should not be set on
  # Tree instances).
  ROOT_INDEX: ClassVar[int] = 0
  NO_PARENT: ClassVar[int] = -1
  UNVISITED: ClassVar[int] = -1

  @property
  def num_actions(self):
    return self.children_index.shape[-1]

  @property
  def num_simulations(self):
    return self.node_visits.shape[-1] - 1

  def qvalues(self, indices):
    """Compute q-values for any node indices in the tree."""
    # pytype: disable=wrong-arg-types  # jnp-type
    if jnp.asarray(indices).shape:
      return jax.vmap(_unbatched_qvalues)(self, indices)
    else:
      return _unbatched_qvalues(self, indices)
    # pytype: enable=wrong-arg-types

  def qvalues_epistemic_variance(self, indices):
    """Compute q-values for any node indices in the tree."""
    # pytype: disable=wrong-arg-types  # jnp-type
    if jnp.asarray(indices).shape:
      return jax.vmap(_unbatched_qvalues_epistemic_variance)(self, indices)
    else:
      return _unbatched_qvalues_epistemic_variance(self, indices)
    # pytype: enable=wrong-arg-types

  def cost_qvalues(self, indices):
    """Compute cost-values for any node indices in the tree."""
    # pytype: disable=wrong-arg-types  # jnp-type
    if jnp.asarray(indices).shape:
      return jax.vmap(_unbatched_cost_qvalues)(self, indices)
    else:
      return _unbatched_cost_qvalues(self, indices)
    # pytype: enable=wrong-arg-types

  def cost_qvalues_epistemic_variance(self, indices):
    """Compute epistemic variance of cost-values for any node indices in the tree."""
    # pytype: disable=wrong-arg-types  # jnp-type
    if jnp.asarray(indices).shape:
      return jax.vmap(_unbatched_cost_qvalues_epistemic_variance)(self, indices)
    else:
      return _unbatched_cost_qvalues_epistemic_variance(self, indices)
    # pytype: enable=wrong-arg-types

  def summary(self) -> SearchSummary:
    """Extract summary statistics for the root node."""
    # Get state and action values for the root nodes.
    chex.assert_rank(self.node_values, 2)
    value = self.node_values[:, EpistemicTree.ROOT_INDEX]
    batch_size, = value.shape
    root_indices = jnp.full((batch_size,), EpistemicTree.ROOT_INDEX)
    qvalues = self.qvalues(root_indices)
    # Extract visit counts and induced probabilities for the root nodes.
    visit_counts = self.children_visits[:, EpistemicTree.ROOT_INDEX].astype(value.dtype)
    total_counts = jnp.sum(visit_counts, axis=-1, keepdims=True)
    visit_probs = visit_counts / jnp.maximum(total_counts, 1)
    visit_probs = jnp.where(total_counts > 0, visit_probs, 1 / self.num_actions)
    # Return relevant stats.
    return SearchSummary(  # pytype: disable=wrong-arg-types  # numpy-scalars
        visit_counts=visit_counts,
        visit_probs=visit_probs,
        value=value,
        qvalues=qvalues)

  def epistemic_summary(self) -> EpistemicSearchSummary:
    """Extract summary statistics for the root node."""
    # Get state and action values for the root nodes.
    chex.assert_rank(self.node_values, 2)
    value = self.node_values[:, EpistemicTree.ROOT_INDEX]
    value_epistemic_std = self.node_values_epistemic_std[:, EpistemicTree.ROOT_INDEX]
    cost_value = self.node_cost_values[:, EpistemicTree.ROOT_INDEX]
    cost_value_epistemic_std = self.node_cost_values_epistemic_std[:, EpistemicTree.ROOT_INDEX]
    batch_size, = value.shape
    root_indices = jnp.full((batch_size,), EpistemicTree.ROOT_INDEX)
    qvalues = self.qvalues(root_indices)
    qvalues_epistemic_variance = self.qvalues_epistemic_variance(root_indices)
    cost_qvalues = self.cost_qvalues(root_indices)
    cost_qvalues_epistemic_variance = self.cost_qvalues_epistemic_variance(root_indices)
    # Extract visit counts and induced probabilities for the root nodes.
    visit_counts = self.children_visits[:, EpistemicTree.ROOT_INDEX].astype(value.dtype)
    total_counts = jnp.sum(visit_counts, axis=-1, keepdims=True)
    visit_probs = visit_counts / jnp.maximum(total_counts, 1)
    visit_probs = jnp.where(total_counts > 0, visit_probs, 1 / self.num_actions)
    # Return relevant stats.
    return EpistemicSearchSummary(  # pytype: disable=wrong-arg-types  # numpy-scalars
      visit_counts=visit_counts,
      visit_probs=visit_probs,
      value=value,
      value_epistemic_std=value_epistemic_std,
      qvalues=qvalues,
      qvalues_epistemic_variance=qvalues_epistemic_variance,
      cost_value=cost_value,
      cost_value_epistemic_std=cost_value_epistemic_std,
      cost_qvalues=cost_qvalues,
      cost_qvalues_epistemic_variance=cost_qvalues_epistemic_variance,
    )

def infer_batch_size(tree: EpistemicTree) -> int:
  """Recovers batch size from `EpistemicTree` data structure."""
  if tree.node_values.ndim != 2:
    raise ValueError("Input tree is not batched.")
  chex.assert_equal_shape_prefix(jax.tree_util.tree_leaves(tree), 1)
  return tree.node_values.shape[0]


# A number of aggregate statistics and predictions are extracted from the
# search data and returned to the user for further processing.
@chex.dataclass(frozen=True)
class EpistemicSearchSummary:
  """Stats from MCTS search."""
  visit_counts: chex.Array
  visit_probs: chex.Array
  value: chex.Array
  value_epistemic_std: chex.Array
  qvalues: chex.Array
  qvalues_epistemic_variance: chex.Array
  cost_value: chex.Array
  cost_value_epistemic_std: chex.Array
  cost_qvalues: chex.Array
  cost_qvalues_epistemic_variance: chex.Array

def _unbatched_qvalues(tree: EpistemicTree, index: int) -> int:
  chex.assert_rank(tree.children_discounts, 2)
  return (  # pytype: disable=bad-return-type  # numpy-scalars
      tree.children_rewards[index]
      + tree.children_discounts[index] * tree.children_values[index]
  )

def _unbatched_qvalues_epistemic_variance(tree: EpistemicTree, index: int) -> int:
  chex.assert_rank(tree.children_discounts, 2)
  return (  # pytype: disable=bad-return-type  # numpy-scalars
      # sigma_r(s,a)^2 + gamma^2 sigma_v(s')^2, assuming independence
      tree.children_rewards_epistemic_variance[index]
      + tree.children_discounts[index] * tree.children_discounts[index] * tree.children_values_epistemic_std[index] * tree.children_values_epistemic_std[index]
  )

def _unbatched_cost_qvalues(tree: EpistemicTree, index: int) -> int:
  chex.assert_rank(tree.children_discounts, 2)
  return (  # pytype: disable=bad-return-type  # numpy-scalars
      tree.children_costs[index]
      + tree.children_discounts[index] * tree.children_cost_values[index]
  )

def _unbatched_cost_qvalues_epistemic_variance(tree: EpistemicTree, index: int) -> int:
  chex.assert_rank(tree.children_discounts, 2)
  return ( 
      # TODO: shouldn't we add the variance of the cost? 
      # gamma^2 * sigma_vc(s')^2 (+ sigma_c(s,a)^2??)
      tree.children_discounts[index]**2 * tree.children_cost_values_epistemic_std[index]**2
  )