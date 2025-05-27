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
"""Tests for Mctx."""

from absl.testing import absltest
import cemctx


class MctxTest(absltest.TestCase):
  """Test emctx can be imported correctly."""

  def test_import(self):
    self.assertTrue(hasattr(cemctx, "gumbel_muzero_policy"))
    self.assertTrue(hasattr(cemctx, "muzero_policy"))
    self.assertTrue(hasattr(cemctx, "qtransform_by_min_max"))
    self.assertTrue(hasattr(cemctx, "qtransform_by_parent_and_siblings"))
    self.assertTrue(hasattr(cemctx, "qtransform_completed_by_mix_value"))
    self.assertTrue(hasattr(cemctx, "PolicyOutput"))
    self.assertTrue(hasattr(cemctx, "RootFnOutput"))
    self.assertTrue(hasattr(cemctx, "RecurrentFnOutput"))
    # Test EMCTS imports
    self.assertTrue(hasattr(cemctx, "epistemic_muzero_policy"))
    self.assertTrue(hasattr(cemctx, "epistemic_gumbel_muzero_policy"))
    self.assertTrue(hasattr(cemctx, "epistemic_qtransform_by_parent_and_siblings"))
    self.assertTrue(hasattr(cemctx, "epistemic_qtransform_completed_by_mix_value"))
    self.assertTrue(hasattr(cemctx, "EpistemicPolicyOutput"))
    self.assertTrue(hasattr(cemctx, "EpistemicRootFnOutput"))
    self.assertTrue(hasattr(cemctx, "EpistemicRecurrentFnOutput"))

if __name__ == "__main__":
  absltest.main()
