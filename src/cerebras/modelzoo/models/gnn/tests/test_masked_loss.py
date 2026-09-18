"""Compare masked losses and gradients with the unpadded PyTorch objective."""

import subprocess
import sys
import unittest

import torch
import torch.nn.functional as F

from cerebras.modelzoo.models.gnn.task.loss import masked_classification_loss


class MaskedLossTests(unittest.TestCase):
    def test_loss_and_gradients_match_selected_targets(self):
        for disable_log_softmax in (False, True):
            for selected in ([True, True, True, True], [True, False, True, False]):
                with self.subTest(ce=disable_log_softmax, selected=selected):
                    logits = torch.tensor(
                        [
                            [1.0, -2.0, 0.5],
                            [0.1, 0.2, 0.3],
                            [-1.0, 2.0, 0.4],
                            [2.0, -1.0, 1.0],
                        ],
                        requires_grad=True,
                    )
                    labels = torch.tensor([2, 1, 0, 2])
                    mask = torch.tensor(selected)
                    expected = F.cross_entropy(logits[mask], labels[mask])
                    expected_grad = torch.autograd.grad(expected, logits)[0]
                    actual = masked_classification_loss(
                        logits,
                        labels,
                        mask,
                        disable_log_softmax=disable_log_softmax,
                    )
                    actual_grad = torch.autograd.grad(actual, logits)[0]
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(actual_grad, expected_grad)

    def test_ignored_labels_and_invalid_padding_are_not_gathered(self):
        for ce in (False, True):
            logits = torch.zeros(4, 2, requires_grad=True)
            loss = masked_classification_loss(
                logits,
                torch.tensor([0, -100, 999, -999]),
                torch.tensor([True, True, False, False]),
                disable_log_softmax=ce,
            )
            torch.testing.assert_close(loss, torch.tensor(2.0).log())
            loss.backward()
            torch.testing.assert_close(
                logits.grad,
                torch.tensor([[-0.5, 0.5], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            )

    def test_empty_supervision_has_zero_loss_and_gradient(self):
        for ce in (False, True):
            logits = torch.randn(4, 2, requires_grad=True)
            loss = masked_classification_loss(
                logits,
                torch.full((4,), -100),
                torch.zeros(4, dtype=torch.bool),
                disable_log_softmax=ce,
            )
            self.assertEqual(loss.dtype, torch.float32)
            self.assertEqual(loss.item(), 0.0)
            loss.backward()
            torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_fp32_count_does_not_overflow_half_precision(self):
        logits = torch.zeros(70000, 2, dtype=torch.float16, requires_grad=True)
        loss = masked_classification_loss(
            logits,
            torch.zeros(70000, dtype=torch.long),
            torch.ones(70000, dtype=torch.bool),
        )
        torch.testing.assert_close(loss, torch.tensor(2.0).log())
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_architecture_import_in_fresh_interpreter(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from cerebras.modelzoo.models.gnn.architectures import GraphSAGE; "
                "from cerebras.modelzoo.models.gnn.task import GNNTaskWrapper",
            ],
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
