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
"""A JAX implementation of batched MCTS."""
import functools
from typing import Any, NamedTuple, Optional, Tuple, TypeVar

import chex
import jax
import jax.numpy as jnp

from cemctx._src import action_selection
from cemctx._src import base
from cemctx._src import epistemic_tree as epistemic_tree_lib

EpistemicTree = epistemic_tree_lib.EpistemicTree
T = TypeVar("T")


def epistemic_search(
    params: base.Params,
    rng_key: chex.PRNGKey,
    *,
    root: base.EpistemicRootFnOutput,
    recurrent_fn: base.EpistemicRecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    num_simulations: int,
    max_depth: Optional[int] = None,
    invalid_actions: Optional[chex.Array] = None,
    extra_data: Any = None,
    loop_fn: base.EpistemicLoopFn = jax.lax.fori_loop) -> EpistemicTree:
  """Performs a full search and returns sampled actions.

  In the shape descriptions, `B` denotes the batch dimension.

  Args:
    params: params to be forwarded to root and recurrent functions.
    rng_key: random number generator state, the key is consumed.
    root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
      `prior_logits` are from a policy network. The shapes are
      `([B, num_actions], [B], [B, ...])`, respectively.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    root_action_selection_fn: function used to select an action at the root.
    interior_action_selection_fn: function used to select an action during
      simulation.
    num_simulations: the number of simulations.
    max_depth: maximum search tree depth allowed during simulation, defined as
      the number of edges from the root to a leaf node.
    invalid_actions: a mask with invalid actions at the root. In the
      mask, invalid actions have ones, and valid actions have zeros.
      Shape `[B, num_actions]`.
    extra_data: extra data passed to `tree.extra_data`. Shape `[B, ...]`.
    loop_fn: Function used to run the simulations. It may be required to pass
      hk.fori_loop if using this function inside a Haiku module.

  Returns:
    `SearchResults` containing outcomes of the search, e.g. `visit_counts`
    `[B, num_actions]`.
  """
  action_selection_fn = action_selection.switching_action_selection_wrapper(
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn
  )

  # Do simulation, expansion, and backward steps.
  batch_size = root.value.shape[0]
  batch_range = jnp.arange(batch_size)
  if max_depth is None:
    max_depth = num_simulations
  if invalid_actions is None:
    invalid_actions = jnp.zeros_like(root.prior_logits)

  def body_fun(sim, loop_state):
    rng_key, tree = loop_state
    rng_key, simulate_key, expand_key = jax.random.split(rng_key, 3)
    # simulate is vmapped and expects batched rng keys.
    simulate_keys = jax.random.split(simulate_key, batch_size)
    parent_index, action = simulate(
        simulate_keys, tree, action_selection_fn, max_depth)
    # A node first expanded on simulation `i`, will have node index `i`.
    # Node 0 corresponds to the root node.
    next_node_index = tree.children_index[batch_range, parent_index, action]
    next_node_index = jnp.where(next_node_index == EpistemicTree.UNVISITED,
                                sim + 1, next_node_index)
    tree = expand(
        params, expand_key, tree, recurrent_fn, parent_index,
        action, next_node_index)
    tree = backward(tree, next_node_index)
    loop_state = rng_key, tree
    return loop_state

  # Allocate all necessary storage.
  tree = instantiate_tree_from_root(root, num_simulations,
                                    root_invalid_actions=invalid_actions,
                                    extra_data=extra_data)
  _, tree = loop_fn(
      0, num_simulations, body_fun, (rng_key, tree))

  return tree


class _SimulationState(NamedTuple):
  """The state for the simulation while loop."""
  rng_key: chex.PRNGKey
  node_index: int
  action: int
  next_node_index: int
  depth: int
  is_continuing: bool


@functools.partial(jax.vmap, in_axes=[0, 0, None, None], out_axes=0)
def simulate(
    rng_key: chex.PRNGKey,
    tree: EpistemicTree,
    action_selection_fn: base.InteriorActionSelectionFn,
    max_depth: int) -> Tuple[chex.Array, chex.Array]:
  """Traverses the tree until reaching an unvisited action or `max_depth`.

  Each simulation starts from the root and keeps selecting actions traversing
  the tree until a leaf or `max_depth` is reached.

  Args:
    rng_key: random number generator state, the key is consumed.
    tree: _unbatched_ MCTS tree state.
    action_selection_fn: function used to select an action during simulation.
    max_depth: maximum search tree depth allowed during simulation.

  Returns:
    `(parent_index, action)` tuple, where `parent_index` is the index of the
    node reached at the end of the simulation, and the `action` is the action to
    evaluate from the `parent_index`.
  """
  def cond_fun(state):
    return state.is_continuing

  def body_fun(state):
    # Preparing the next simulation state.
    node_index = state.next_node_index
    rng_key, action_selection_key = jax.random.split(state.rng_key)
    action = action_selection_fn(action_selection_key, tree, node_index,
                                 state.depth)
    next_node_index = tree.children_index[node_index, action]
    # The returned action will be visited.
    depth = state.depth + 1
    is_before_depth_cutoff = depth < max_depth
    is_visited = next_node_index != EpistemicTree.UNVISITED
    is_continuing = jnp.logical_and(is_visited, is_before_depth_cutoff)
    return _SimulationState(  # pytype: disable=wrong-arg-types  # jax-types
        rng_key=rng_key,
        node_index=node_index,
        action=action,
        next_node_index=next_node_index,
        depth=depth,
        is_continuing=is_continuing)

  node_index = jnp.array(EpistemicTree.ROOT_INDEX, dtype=jnp.int32)
  depth = jnp.zeros((), dtype=tree.children_prior_logits.dtype)
  # pytype: disable=wrong-arg-types  # jnp-type
  initial_state = _SimulationState(
      rng_key=rng_key,
      node_index=tree.NO_PARENT,
      action=tree.NO_PARENT,
      next_node_index=node_index,
      depth=depth,
      is_continuing=jnp.array(True))
  # pytype: enable=wrong-arg-types
  end_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)

  # Returning a node with a selected action.
  # The action can be already visited, if the max_depth is reached.
  return end_state.node_index, end_state.action


def expand(
    params: chex.Array,
    rng_key: chex.PRNGKey,
    tree: EpistemicTree[T],
    recurrent_fn: base.EpistemicRecurrentFn,
    parent_index: chex.Array,
    action: chex.Array,
    next_node_index: chex.Array) -> EpistemicTree[T]:
  """Create and evaluate child nodes from given nodes and unvisited actions.

  Args:
    params: params to be forwarded to recurrent function.
    rng_key: random number generator state.
    tree: the MCTS tree state to update.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    parent_index: the index of the parent node, from which the action will be
      expanded. Shape `[B]`.
    action: the action to expand. Shape `[B]`.
    next_node_index: the index of the newly expanded node. This can be the index
      of an existing node, if `max_depth` is reached. Shape `[B]`.

  Returns:
    tree: updated MCTS tree state.
  """
  batch_size = epistemic_tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  chex.assert_shape([parent_index, action, next_node_index], (batch_size,))

  # Retrieve states for nodes to be evaluated.
  embedding = jax.tree.map(
      lambda x: x[batch_range, parent_index], tree.embeddings)

  # Evaluate and create a new node.
  step, embedding = recurrent_fn(params, rng_key, action, embedding)
  chex.assert_shape(step.prior_logits, [batch_size, tree.num_actions])
  chex.assert_shape(step.reward, [batch_size])
  chex.assert_shape(step.discount, [batch_size])
  chex.assert_shape(step.value, [batch_size])
  chex.assert_shape(step.value_epistemic_variance, [batch_size])

  # Calculate discounted cumulative cost from root to the new node.
  # TODO: should we also account for uncertainty here?
  parent_cumulative_cost = tree.node_cumulative_costs[batch_range, parent_index]
  cumulative_cost = parent_cumulative_cost + step.discount * step.cost

  tree = update_tree_node(
      tree, next_node_index, step.prior_logits, step.value, step.value_epistemic_variance, 
      cumulative_cost, step.cost_value, step.cost_value_epistemic_variance, embedding)

  # Return updated tree topology.
  return tree.replace(
      children_index=batch_update(
          tree.children_index, next_node_index, parent_index, action),
      children_rewards=batch_update(
          tree.children_rewards, step.reward, parent_index, action),
      children_discounts=batch_update(
          tree.children_discounts, step.discount, parent_index, action),
      parents=batch_update(tree.parents, parent_index, next_node_index),
      action_from_parent=batch_update(
          tree.action_from_parent, action, next_node_index),
      children_rewards_epistemic_variance=batch_update(
          tree.children_rewards_epistemic_variance, step.reward_epistemic_variance,
          parent_index, action),
      children_costs=batch_update(
          tree.children_costs, step.cost, parent_index, action),
      children_costs_epistemic_variance=batch_update(
          tree.children_costs_epistemic_variance, step.cost_epistemic_variance,
          parent_index, action)
  )


@jax.vmap
def backward(
    tree: EpistemicTree[T],
    leaf_index: chex.Numeric) -> EpistemicTree[T]:
  """Goes up and updates the tree until all nodes reached the root.

  Args:
    tree: the MCTS tree state to update, without the batch size.
    leaf_index: the node index from which to do the backward.

  Returns:
    Updated MCTS tree state.
  """

  def cond_fun(loop_state):
    _, _, _, index = loop_state
    return index != EpistemicTree.ROOT_INDEX

  def body_fun(loop_state):
    # Here we update the value of our parent, so we start by reversing.
    tree, leaf_value, leaf_value_epistemic_variance, leaf_cost_value, leaf_cost_value_epistemic_variance, index = loop_state
    parent = tree.parents[index]
    count = tree.node_visits[parent]
    action = tree.action_from_parent[index]
    reward = tree.children_rewards[parent, action]
    leaf_value = reward + tree.children_discounts[parent, action] * leaf_value
    parent_value = (
        tree.node_values[parent] * count + leaf_value) / (count + 1.0)
    children_values = tree.node_values[index]
    children_counts = tree.children_visits[parent, action] + 1

    # Propagate the uncertainty of rewards
    reward_epistemic_variance = tree.children_rewards_epistemic_variance[parent, action]
    leaf_value_epistemic_variance = reward_epistemic_variance + tree.children_discounts[parent, action] * tree.children_discounts[parent, action] * leaf_value_epistemic_variance
    # Note that the leaf and the reward uncertainties are variances (sigma^2), but the averaged saved is an std (\sqrt(sigma^2))
    parent_value_epistemic_std = (tree.node_values_epistemic_std[parent] * count + jnp.sqrt(leaf_value_epistemic_variance)) / (count + 1.0)
    children_values_epistemic_std = tree.node_values_epistemic_std[index]

    # Cost values
    cost = tree.children_costs[parent, action]
    leaf_cost_value = cost + tree.children_discounts[parent, action] * leaf_cost_value
    parent_cost_value = (
        tree.node_cost_values[parent] * count + leaf_cost_value) / (count + 1.0)
    children_cost_values = tree.node_cost_values[index]

    # Propagate the uncertainty of costs
    cost_epistemic_variance = tree.children_costs_epistemic_variance[parent, action]
    leaf_cost_value_epistemic_variance = cost_epistemic_variance + tree.children_discounts[parent, action] * tree.children_discounts[parent, action] * leaf_cost_value_epistemic_variance
    parent_cost_value_epistemic_std = (tree.node_cost_values_epistemic_std[parent] * count + jnp.sqrt(leaf_cost_value_epistemic_variance)) / (count + 1.0)
    children_cost_values_epistemic_std = tree.node_cost_values_epistemic_std[index]

    tree = tree.replace(
        node_values=update(tree.node_values, parent_value, parent),
        node_values_epistemic_std=update(tree.node_values_epistemic_std, parent_value_epistemic_std, parent),
        node_visits=update(tree.node_visits, count + 1, parent),
        children_values=update(
            tree.children_values, children_values, parent, action),
        children_values_epistemic_std=update(
            tree.children_values_epistemic_std, children_values_epistemic_std, parent, action),
        children_visits=update(
            tree.children_visits, children_counts, parent, action),
        node_cost_values=update(
            tree.node_cost_values, parent_cost_value, parent),
        node_cost_values_epistemic_std=update(
            tree.node_cost_values_epistemic_std, parent_cost_value_epistemic_std, parent),
        children_cost_values=update(
            tree.children_cost_values, children_cost_values, parent, action),
        children_cost_values_epistemic_std=update(
            tree.children_cost_values_epistemic_std, children_cost_values_epistemic_std, parent, action)
    )

    return tree, leaf_value, leaf_value_epistemic_variance, leaf_cost_value, leaf_cost_value_epistemic_variance, parent

  leaf_index = jnp.asarray(leaf_index, dtype=jnp.int32)
  loop_state = (tree, tree.node_values[leaf_index], jnp.square(tree.node_values_epistemic_std[leaf_index]), tree.node_cost_values[leaf_index], jnp.square(tree.node_cost_values_epistemic_std[leaf_index]), leaf_index)
  tree, _, _, _ = jax.lax.while_loop(cond_fun, body_fun, loop_state)

  return tree


# Utility function to set the values of certain indices to prescribed values.
# This is vmapped to operate seamlessly on batches.
def update(x, vals, *indices):
  return x.at[indices].set(vals)


batch_update = jax.vmap(update)


def update_tree_node(
    tree: EpistemicTree[T],
    node_index: chex.Array,
    prior_logits: chex.Array,
    value: chex.Array,
    value_epistemic_variance: chex.Array,
    cumulative_cost: chex.Array,
    cost_value: chex.Array,
    cost_value_epistemic_variance: chex.Array,
    embedding: chex.Array) -> EpistemicTree[T]:
  """Updates the tree at node index.

  Args:
    tree: `Tree` to whose node is to be updated.
    node_index: the index of the expanded node. Shape `[B]`.
    prior_logits: the prior logits to fill in for the new node, of shape
      `[B, num_actions]`.
    value: the value to fill in for the new node. Shape `[B]`.
    value_epistemic_variance: the epistemic variance of the value to fill in the new node. Shape '[B]'.
    cost_value: the cost value to fill in for the new node. Shape `[B]`.
    cost_value_epistemic_variance: the epistemic variance of the cost value to fill in the new node. Shape '[B]'.
    embedding: the state embeddings for the node. Shape `[B, ...]`.

  Returns:
    The new tree with updated nodes.
  """
  batch_size = epistemic_tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  chex.assert_shape(prior_logits, (batch_size, tree.num_actions))

  # When using max_depth, a leaf can be expanded multiple times.
  new_visit = tree.node_visits[batch_range, node_index] + 1
  updates = dict(  # pylint: disable=use-dict-literal
      children_prior_logits=batch_update(
          tree.children_prior_logits, prior_logits, node_index),
      raw_values=batch_update(
          tree.raw_values, value, node_index),
      raw_values_epistemic_variance=batch_update(
          tree.raw_values_epistemic_variance, value_epistemic_variance, node_index),
      node_values=batch_update(
          tree.node_values, value, node_index),
      node_values_epistemic_std=batch_update(
          # Note that becaue the epistemic-node-"variance" is saved as std, we sqrt the variance
          tree.node_values_epistemic_std, jnp.sqrt(value_epistemic_variance), node_index),
      raw_cost_values=batch_update(
          tree.raw_cost_values, cost_value, node_index),
      raw_cost_values_epistemic_variance=batch_update(
          tree.raw_cost_values_epistemic_variance, cost_value_epistemic_variance, node_index),
      node_cost_values=batch_update(
          tree.node_cost_values, cost_value, node_index),
      node_cost_values_epistemic_std=batch_update(
          tree.node_cost_values_epistemic_std, jnp.sqrt(cost_value_epistemic_variance), node_index),
      node_visits=batch_update(
          tree.node_visits, new_visit, node_index),
      node_cumulative_costs=batch_update(
          tree.node_cumulative_costs, cumulative_cost, node_index),
      embeddings=jax.tree.map(
          lambda t, s: batch_update(t, s, node_index),
          tree.embeddings, embedding)
  )

  return tree.replace(**updates)


def instantiate_tree_from_root(
    root: base.EpistemicRootFnOutput,
    num_simulations: int,
    root_invalid_actions: chex.Array,
    extra_data: Any) -> EpistemicTree:
  """Initializes tree state at search root."""
  chex.assert_rank(root.prior_logits, 2)
  batch_size, num_actions = root.prior_logits.shape
  chex.assert_shape(root.value, [batch_size])
  num_nodes = num_simulations + 1
  data_dtype = root.value.dtype
  beta_v = root.beta_v
  beta_c = root.beta_c
  cost_threshold = root.cost_threshold
  batch_node = (batch_size, num_nodes)
  batch_node_action = (batch_size, num_nodes, num_actions)

  def _zeros(x):
    return jnp.zeros(batch_node + x.shape[1:], dtype=x.dtype)

  # Create a new empty tree state and fill its root.
  tree = EpistemicTree(
      node_visits=jnp.zeros(batch_node, dtype=jnp.int32),
      raw_values=jnp.zeros(batch_node, dtype=data_dtype),
      raw_values_epistemic_variance=jnp.zeros(batch_node, dtype=data_dtype),
      node_values=jnp.zeros(batch_node, dtype=data_dtype),
      node_values_epistemic_std=jnp.zeros(batch_node, dtype=data_dtype),
      parents=jnp.full(batch_node, EpistemicTree.NO_PARENT, dtype=jnp.int32),
      action_from_parent=jnp.full(
          batch_node, EpistemicTree.NO_PARENT, dtype=jnp.int32),
      children_index=jnp.full(
          batch_node_action, EpistemicTree.UNVISITED, dtype=jnp.int32),
      children_prior_logits=jnp.zeros(
          batch_node_action, dtype=root.prior_logits.dtype),
      children_values=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_values_epistemic_std=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_visits=jnp.zeros(batch_node_action, dtype=jnp.int32),
      children_rewards=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_rewards_epistemic_variance=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_discounts=jnp.zeros(batch_node_action, dtype=data_dtype),
      embeddings=jax.tree.map(_zeros, root.embedding),
      root_invalid_actions=root_invalid_actions,
      extra_data=extra_data,
      beta_v=beta_v,
      raw_cost_values=jnp.zeros(batch_node, dtype=data_dtype),
      node_cost_values=jnp.zeros(batch_node, dtype=data_dtype),
      children_costs=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_cost_values=jnp.zeros(batch_node_action, dtype=data_dtype),
      raw_cost_values_epistemic_variance=jnp.zeros(batch_node, dtype=data_dtype),
      node_cost_values_epistemic_std=jnp.zeros(batch_node, dtype=data_dtype),
      children_costs_epistemic_variance=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_cost_values_epistemic_std=jnp.zeros(batch_node_action, dtype=data_dtype),
      beta_c=beta_c,
      cost_threshold=cost_threshold,
  )

  root_index = jnp.full([batch_size], EpistemicTree.ROOT_INDEX)
  root_cumulative_cost = jnp.zeros(batch_size, dtype=data_dtype)
  tree = update_tree_node(
      tree, root_index, root.prior_logits, root.value, root.value_epistemic_variance, 
      root_cumulative_cost, root.cost_value, root.cost_value_epistemic_variance, root.embedding)
  return tree