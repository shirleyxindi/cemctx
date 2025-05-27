import unittest
import jax.numpy as jnp
import chex

from cemctx._src.epistemic_tree import EpistemicTree, _unbatched_qvalues_epistemic_variance, EpistemicSearchSummary

class TestEpistemicTree(unittest.TestCase):

    def setUp(self):
        # Initialize an example EpistemicTree for testing
        self.tree = EpistemicTree(
            node_visits=jnp.array([[10, 20]]),
            raw_values=jnp.array([[1.0, 2.0]]),
            node_values=jnp.array([[1.0, 2.0]]),
            parents=jnp.array([[0, 1]]),
            action_from_parent=jnp.array([[0, 1]]),
            children_index=jnp.array([[[0, 1], [1, 0]]]),
            children_prior_logits=jnp.array([[[0.1, 0.9], [0.8, 0.2]]]),
            children_visits=jnp.array([[[1, 2], [3, 4]]]),
            children_rewards=jnp.array([[1.0, 2.0]]),  # [N, num_actions]
            children_discounts=jnp.array([[0.9, 0.9]]),  # [N, num_actions]
            children_values=jnp.array([[1.0, 2.0]]),  # [N, num_actions]
            embeddings=None,
            root_invalid_actions=jnp.array([[0, 1]]),
            extra_data=None,
            raw_values_epistemic_variance=jnp.array([[0.1, 0.2]]),
            node_values_epistemic_std=jnp.array([[0.1, 0.2]]),
            children_rewards_epistemic_variance=jnp.array([[0.01, 0.02]]),  # [N, num_actions]
            children_values_epistemic_std=jnp.array([[0.1, 0.2]]),  # [N, num_actions]
            beta=jnp.array([0.5])
        )

    def test_qvalues_epistemic_variance_single(self):
        # Test qvalues_epistemic_variance with a single index
        result = self.tree.qvalues_epistemic_variance(1)
        expected = jnp.array([0.01 + 0.9 ** 2 * 0.1 ** 2, 0.02 + 0.9 ** 2 * 0.2 ** 2])
        # Ensure correct values are being tested
        chex.assert_trees_all_close(result, expected)

    def test_epistemic_summary(self):
        # Test epistemic_summary method
        summary = self.tree.epistemic_summary()
        self.assertIsInstance(summary, EpistemicSearchSummary)

        expected_qvalues = self.tree.qvalues(jnp.array([0]))
        expected_qvalues_epistemic_variance = self.tree.qvalues_epistemic_variance(jnp.array([0]))
        expected_visit_counts = jnp.array([1, 2])
        expected_visit_probs = jnp.array([1/3, 2/3])

        chex.assert_trees_all_close(summary.qvalues, expected_qvalues)
        chex.assert_trees_all_close(summary.qvalues_epistemic_variance, expected_qvalues_epistemic_variance)
        chex.assert_trees_all_close(summary.visit_counts, expected_visit_counts)
        chex.assert_trees_all_close(summary.visit_probs, expected_visit_probs)
        chex.assert_trees_all_close(summary.value, jnp.array([1.0]))

class TestUnbatchedQValuesEpistemicVariance(unittest.TestCase):
    def test_unbatched_qvalues_epistemic_variance(self):
        # Create a minimal EpistemicTree instance with relevant fields
        tree = EpistemicTree(
            node_visits=None,
            raw_values=None,
            node_values=None,
            parents=None,
            action_from_parent=None,
            children_index=None,
            children_prior_logits=None,
            children_visits=None,
            children_rewards=jnp.array([[1.0, 2.0]]),
            children_discounts=jnp.array([[0.9, 0.9]]),
            children_values=None,
            embeddings=None,
            root_invalid_actions=None,
            extra_data=None,
            raw_values_epistemic_variance=None,
            node_values_epistemic_std=None,
            children_rewards_epistemic_variance=jnp.array([[0.1, 0.2]]),
            children_values_epistemic_std=jnp.array([[0.1, 0.2]]),
            beta=None
        )

        result = _unbatched_qvalues_epistemic_variance(tree, 0)
        expected = jnp.array([0.1 + 0.9**2 * 0.1**2, 0.2 + 0.9**2 * 0.2**2])
        chex.assert_trees_all_close(result, expected)

if __name__ == '__main__':
    unittest.main()
