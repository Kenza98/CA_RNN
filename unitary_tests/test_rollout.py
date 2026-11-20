"""Unit tests for src/tests/rollout.py, on tiny hand-built grids.

Run from the repo root:
    python -m unittest unitary_tests.test_rollout -v
"""

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
import xarray as xr

from src.data.extractors import extract_neighborhood, extract_pointwise
from src.models.gru import GRU
from src.models.lstm import LSTM
from src.tests import rollout

NAN = float("nan")


# Toy models: deterministic, so the expected outputs can be written by hand.

class Persistence(nn.Module):
    """Returns the centre cell of the last time step: next grid == last grid."""

    def forward(self, x):  # x: (n_cells, seq_length, F)
        centre = x.shape[-1] // 2
        return x[:, -1, centre : centre + 1]


class Constant(nn.Module):
    """Returns the same (standardised) value for every cell."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full((x.shape[0], 1), self.value)


class PlusOne(nn.Module):
    """Last centre value + 1 (in standardised units)."""

    def forward(self, x):
        centre = x.shape[-1] // 2
        return x[:, -1, centre : centre + 1] + 1.0


class NanModel(nn.Module):
    """Returns NaN everywhere, which predict_next_grid must reject."""

    def forward(self, x):
        return torch.full((x.shape[0], 1), NAN)


class PointwiseOnly(nn.Module):
    """Persistence that fails unless it gets one feature per cell (the input without CA)."""

    def forward(self, x):
        assert x.shape[-1] == 1, f"expected pointwise input, got {x.shape[-1]} features"
        return x[:, -1, :]


def ramp_grid(H=5, W=5):
    """(H, W) grid with distinct values 0..H*W-1."""
    return torch.arange(H * W, dtype=torch.float32).view(H, W)


class TestLoadSstSeries(unittest.TestCase):
    def _write_nc(self, path, with_depth):
        data = np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4)
        data[0, 0, 0] = np.nan  # a land cell
        if with_depth:
            da = xr.DataArray(
                np.stack([data, data + 100]), dims=("depth", "time", "lat", "lon")
            ).transpose("time", "depth", "lat", "lon")
        else:
            da = xr.DataArray(data, dims=("time", "lat", "lon"))
        da.to_dataset(name="thetao").to_netcdf(path)
        return data

    def test_without_depth(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sst_test.nc"
            data = self._write_nc(path, with_depth=False)
            with mock.patch.object(rollout, "raw_path", return_value=path):
                sst = rollout.load_sst_series("test")
        self.assertEqual(sst.dtype, torch.float32)
        self.assertEqual(tuple(sst.shape), (3, 2, 4))
        torch.testing.assert_close(sst, torch.from_numpy(data), equal_nan=True)

    def test_keeps_surface_depth_only(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sst_test.nc"
            data = self._write_nc(path, with_depth=True)
            with mock.patch.object(rollout, "raw_path", return_value=path):
                sst = rollout.load_sst_series("test")
        self.assertEqual(tuple(sst.shape), (3, 2, 4))
        torch.testing.assert_close(sst, torch.from_numpy(data), equal_nan=True)


class TestLoadCheckpoint(unittest.TestCase):
    def test_rebuilds_model_and_stats(self):
        torch.manual_seed(0)
        original = GRU(9, 4, 1, num_layers=2)
        ckpt = {
            "model_type": "GRU",
            "dataset": "nca",
            "hidden_dim": 4,
            "num_layers": 2,
            "global_mean": 15.5,
            "global_std": 2.0,
            "GRUStateDict": original.state_dict(),
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tiny.pt"
            torch.save(ckpt, path)
            model, mean, std, meta = rollout.load_checkpoint(path, "cpu")

        self.assertIsInstance(model, GRU)
        self.assertFalse(model.training)
        self.assertEqual((mean, std), (15.5, 2.0))
        self.assertIsInstance(mean, float)
        self.assertNotIn("GRUStateDict", meta)
        self.assertEqual(meta["hidden_dim"], 4)

        x = torch.randn(3, 6, 9)
        with torch.no_grad():
            torch.testing.assert_close(model(x), original.eval()(x))


class TestPredictNextGrid(unittest.TestCase):
    def test_persistence_all_sea(self):
        """5x5 all-sea grid: the 3x3 interior is predicted, the ring is NaN."""
        window = torch.stack([ramp_grid() + t for t in range(3)])
        pred, mask = rollout.predict_next_grid(
            Persistence(), window, extract_neighborhood, mean=10.0, std=4.0
        )
        expected_mask = torch.zeros(5, 5, dtype=torch.bool)
        expected_mask[1:-1, 1:-1] = True
        self.assertTrue(torch.equal(mask, expected_mask))
        torch.testing.assert_close(pred[1:-1, 1:-1], window[-1, 1:-1, 1:-1])
        self.assertTrue(torch.isnan(pred[~expected_mask]).all())

    def test_destandardisation(self):
        """A model output of 0.5 maps back to mean + 0.5 * std."""
        window = torch.zeros(2, 4, 4)
        pred, mask = rollout.predict_next_grid(
            Constant(0.5), window, extract_neighborhood, mean=20.0, std=2.0
        )
        torch.testing.assert_close(pred[mask], torch.full((4,), 21.0))

    def test_land_masks_neighbours(self):
        """A land cell at (2, 2) of a 5x5 grid makes all 9 interior cells unpredictable,
        and a land cell at (1, 1) blocks only the 4 interior cells touching it."""
        window = ramp_grid().expand(2, 5, 5).clone()
        window[:, 2, 2] = NAN
        _, mask = rollout.predict_next_grid(
            Persistence(), window, extract_neighborhood, 0.0, 1.0
        )
        self.assertFalse(mask.any())

        window = ramp_grid().expand(2, 5, 5).clone()
        window[:, 1, 1] = NAN
        pred, mask = rollout.predict_next_grid(
            Persistence(), window, extract_neighborhood, 0.0, 1.0
        )
        self.assertEqual(
            mask.nonzero().tolist(), [[1, 3], [2, 3], [3, 1], [3, 2], [3, 3]]
        )
        torch.testing.assert_close(pred[mask], window[-1][mask])

    def test_pointwise_extractor_same_cells(self):
        """extract_pointwise uses the same 3x3 validity criterion."""
        window = ramp_grid().expand(2, 5, 5).clone()
        window[:, 1, 1] = NAN
        pred, mask = rollout.predict_next_grid(
            Persistence(), window, extract_pointwise, 0.0, 1.0
        )
        self.assertEqual(
            mask.nonzero().tolist(), [[1, 3], [2, 3], [3, 1], [3, 2], [3, 3]]
        )
        torch.testing.assert_close(pred[mask], window[-1][mask])

    def test_nan_output_raises(self):
        window = torch.zeros(2, 4, 4)
        with self.assertRaises(AssertionError):
            rollout.predict_next_grid(NanModel(), window, extract_neighborhood, 0.0, 1.0)


class TestFillWithGroundTruth(unittest.TestCase):
    def test_fills_only_unpredicted_sea(self):
        pred = torch.tensor([[NAN, NAN, NAN], [NAN, 5.0, NAN], [NAN, NAN, NAN]])
        predicted = ~torch.isnan(pred)
        sea = torch.ones(3, 3, dtype=torch.bool)
        sea[0, 0] = False  # land
        truth = torch.arange(9, dtype=torch.float32).view(3, 3)

        grid = rollout.fill_with_ground_truth(pred, predicted, sea, None, truth)

        self.assertEqual(grid[1, 1].item(), 5.0)  # model value kept, not truth (4)
        self.assertTrue(math.isnan(grid[0, 0].item()))  # land stays NaN
        fill = sea & ~predicted
        torch.testing.assert_close(grid[fill], truth[fill])
        self.assertTrue(torch.isnan(pred[0, 1]))  # input not modified in place


class TestFillWithNeighbourMean(unittest.TestCase):
    def test_single_pass(self):
        """Centre predicted as 8: the ring cells all see only it, so they take 8."""
        pred = torch.full((3, 3), NAN)
        pred[1, 1] = 8.0
        predicted = ~torch.isnan(pred)
        sea = torch.ones(3, 3, dtype=torch.bool)
        window = torch.zeros(1, 3, 3)

        grid = rollout.fill_with_neighbour_mean(pred, predicted, sea, window)
        torch.testing.assert_close(grid, torch.full((3, 3), 8.0))

    def test_mean_of_available_neighbours_and_land(self):
        """(0, 0) is land; (0, 1) sees (1, 0)=2, (1, 1)=4 and (1, 2)=6 -> mean 4."""
        pred = torch.tensor([[NAN, NAN, NAN], [2.0, 4.0, 6.0], [2.0, 4.0, 6.0]])
        predicted = ~torch.isnan(pred)
        sea = torch.ones(3, 3, dtype=torch.bool)
        sea[0, 0] = False
        window = torch.zeros(1, 3, 3)

        grid = rollout.fill_with_neighbour_mean(pred, predicted, sea, window)
        self.assertTrue(math.isnan(grid[0, 0].item()))
        self.assertAlmostEqual(grid[0, 1].item(), 4.0)  # (2 + 4 + 6) / 3
        self.assertAlmostEqual(grid[0, 2].item(), 5.0)  # (4 + 6) / 2
        torch.testing.assert_close(grid[1:], pred[1:])  # predicted cells untouched

    def test_multiple_passes(self):
        """1x4 strip with only (0, 0) predicted: values propagate one cell per pass."""
        pred = torch.tensor([[1.0, NAN, NAN, NAN]])
        predicted = ~torch.isnan(pred)
        sea = torch.ones(1, 4, dtype=torch.bool)
        window = torch.full((1, 1, 4), -99.0)

        grid = rollout.fill_with_neighbour_mean(pred, predicted, sea, window, max_iter=4)
        torch.testing.assert_close(grid, torch.ones(1, 4))

    def test_fallback_to_previous_value(self):
        """With max_iter=1 only (0, 1) gets filled; the rest falls back to window[-1]."""
        pred = torch.tensor([[1.0, NAN, NAN, NAN]])
        predicted = ~torch.isnan(pred)
        sea = torch.ones(1, 4, dtype=torch.bool)
        window = torch.stack([torch.zeros(1, 4), torch.tensor([[0.0, 0.0, 7.0, 9.0]])])

        grid = rollout.fill_with_neighbour_mean(pred, predicted, sea, window, max_iter=1)
        torch.testing.assert_close(grid, torch.tensor([[1.0, 1.0, 7.0, 9.0]]))

    def test_isolated_cell_falls_back(self):
        """A sea cell surrounded only by land never gets a neighbour mean."""
        pred = torch.full((3, 3), NAN)
        predicted = torch.zeros(3, 3, dtype=torch.bool)
        sea = torch.zeros(3, 3, dtype=torch.bool)
        sea[1, 1] = True
        window = torch.full((2, 3, 3), 3.0)

        grid = rollout.fill_with_neighbour_mean(pred, predicted, sea, window)
        self.assertEqual(grid[1, 1].item(), 3.0)
        self.assertEqual(torch.isnan(grid).sum().item(), 8)


class TestRolloutCaRnn(unittest.TestCase):
    def setUp(self):
        # 10 frames of a 5x5 grid, each frame = ramp + t, with land at (0, 0)
        self.sst = torch.stack([ramp_grid() + t for t in range(10)])
        self.sst[:, 0, 0] = NAN

    def test_shapes_and_truth_slice(self):
        preds, truth, masks = rollout.rollout_ca_rnn(
            Persistence(), self.sst, 0.0, 1.0,
            rollout.fill_with_ground_truth, start=1, n_steps=3, seq_length=4,
        )
        self.assertEqual(tuple(preds.shape), (3, 5, 5))
        self.assertEqual(tuple(masks.shape), (3, 5, 5))
        self.assertEqual(masks.dtype, torch.bool)
        torch.testing.assert_close(truth, self.sst[5:8], equal_nan=True)

    def test_persistence_is_constant_on_predicted_cells(self):
        """Persistence repeats the last seed frame on the interior at every step."""
        preds, _, masks = rollout.rollout_ca_rnn(
            Persistence(), self.sst, 0.0, 1.0,
            rollout.fill_with_neighbour_mean, start=0, n_steps=3, seq_length=4,
        )
        last_seed = self.sst[3]
        for k in range(3):
            m = masks[k]
            torch.testing.assert_close(preds[k][m], last_seed[m])
        # land is never filled
        self.assertTrue(torch.isnan(preds[:, 0, 0]).all())

    def test_predictions_are_fed_back(self):
        """PlusOne with std=1 adds 1 degC per step: step k gives last_seed + k + 1,
        which only holds if each prediction is appended to the window."""
        preds, _, masks = rollout.rollout_ca_rnn(
            PlusOne(), self.sst, 0.0, 1.0,
            rollout.fill_with_neighbour_mean, start=0, n_steps=4, seq_length=3,
        )
        last_seed = self.sst[2]
        for k in range(4):
            m = masks[k]
            self.assertTrue(m.any())
            torch.testing.assert_close(preds[k][m], last_seed[m] + k + 1)

    def test_ground_truth_fill_uses_truth_on_ring(self):
        preds, truth, masks = rollout.rollout_ca_rnn(
            Persistence(), self.sst, 0.0, 1.0,
            rollout.fill_with_ground_truth, start=0, n_steps=2, seq_length=4,
        )
        sea = ~torch.isnan(self.sst[0])
        for k in range(2):
            fill = sea & ~masks[k]
            torch.testing.assert_close(preds[k][fill], truth[k][fill])

    def test_too_few_frames_raises(self):
        with self.assertRaises(ValueError):
            rollout.rollout_ca_rnn(
                Persistence(), self.sst, 0.0, 1.0,
                rollout.fill_with_ground_truth, start=5, n_steps=3, seq_length=4,
            )

    def test_input_series_not_modified(self):
        before = self.sst.clone()
        rollout.rollout_ca_rnn(
            PlusOne(), self.sst, 0.0, 1.0,
            rollout.fill_with_neighbour_mean, start=0, n_steps=3, seq_length=3,
        )
        torch.testing.assert_close(self.sst, before, equal_nan=True)


class TestRolloutRnn(unittest.TestCase):
    def setUp(self):
        # 10 frames of a 5x5 grid with distinct values, land at (0, 0)
        self.sst = torch.stack([ramp_grid() * (1 + 0.1 * t) for t in range(10)])
        self.sst[:, 0, 0] = NAN

    def _run(self, model, fill_fn=rollout.fill_with_neighbour_mean, start=0, n_steps=3):
        """rollout_rnn on self.sst, with mean=0, std=1 and a 4-frame window."""
        return rollout.rollout_rnn(
            model,
            self.sst,
            mean=0.0,
            std=1.0,
            fill_fn=fill_fn,
            start=start,
            n_steps=n_steps,
            seq_length=4,
        )

    def test_shapes_and_truth_slice(self):
        preds, truth, masks = self._run(Persistence(), start=2)
        self.assertEqual(tuple(preds.shape), (3, 5, 5))
        self.assertEqual(masks.dtype, torch.bool)
        torch.testing.assert_close(truth, self.sst[6:9], equal_nan=True)

    def test_uses_pointwise_input(self):
        """PointwiseOnly fails on anything but 1 feature per cell, then repeats the last frame."""
        preds, _, masks = self._run(PointwiseOnly())
        for k in range(3):
            torch.testing.assert_close(preds[k][masks[k]], self.sst[3][masks[k]])

    def test_predictions_are_fed_back(self):
        """PlusOne adds 1 degC per step: step k gives last_seed + k + 1."""
        preds, _, masks = self._run(PlusOne(), n_steps=4)
        for k in range(4):
            torch.testing.assert_close(preds[k][masks[k]], self.sst[3][masks[k]] + k + 1)

    def test_same_cells_as_ca_rnn(self):
        sst_with_coast = self.sst.clone()
        sst_with_coast[:, 2, 3] = NAN
        fill = rollout.fill_with_neighbour_mean

        _, _, masks_rnn = rollout.rollout_rnn(
            Persistence(), sst_with_coast, 0.0, 1.0, fill, start=1, n_steps=3, seq_length=4
        )
        _, _, masks_ca = rollout.rollout_ca_rnn(
            Persistence(), sst_with_coast, 0.0, 1.0, fill, start=1, n_steps=3, seq_length=4
        )
        self.assertTrue(torch.equal(masks_rnn, masks_ca))

    def test_fill_does_not_change_predicted_cells(self):
        """Without CA, filled cells never enter a predicted cell's input."""
        preds_truth_fill, _, masks = self._run(PlusOne(), rollout.fill_with_ground_truth, n_steps=4)
        preds_mean_fill, _, masks_mean_fill = self._run(PlusOne(), rollout.fill_with_neighbour_mean, n_steps=4)
        self.assertTrue(torch.equal(masks, masks_mean_fill))

        # same values on the predicted cells...
        torch.testing.assert_close(preds_truth_fill[masks], preds_mean_fill[masks])

        # ...although the filled cells differ between the two fills
        filled = ~masks & ~torch.isnan(preds_truth_fill)
        self.assertFalse(torch.allclose(preds_truth_fill[filled], preds_mean_fill[filled]))

    def test_real_lstm_runs(self):
        """A 1-input LSTM, as built by load_checkpoint for an 'nn' checkpoint."""
        torch.manual_seed(0)
        model = LSTM(rollout.INPUT_DIMS["nn"], 4, 1).eval()
        preds, _, masks = self._run(model)
        self.assertTrue(torch.isfinite(preds[masks]).all())

    def test_too_few_frames_raises(self):
        with self.assertRaises(ValueError):
            self._run(Persistence(), start=5)


class TestRolloutNeighbourAverage(unittest.TestCase):
    def test_one_step_by_hand(self):
        """3x3 all-sea grid: the centre becomes the mean of the 9 values of the last frame."""
        last = torch.arange(9, dtype=torch.float32).view(3, 3)
        sst = torch.stack([torch.zeros(3, 3), last, torch.zeros(3, 3)])
        preds, _, masks = rollout.rollout_neighbour_average(
            sst, rollout.fill_with_ground_truth, start=0, n_steps=1, seq_length=2
        )
        self.assertEqual(masks[0].nonzero().tolist(), [[1, 1]])
        self.assertAlmostEqual(preds[0, 1, 1].item(), 4.0)  # (0 + ... + 8) / 9

    def test_uniform_field_is_fixed_point(self):
        """A constant field stays constant, including the filled ring."""
        sst = torch.full((8, 5, 5), 12.5)
        preds, _, masks = rollout.rollout_neighbour_average(
            sst, rollout.fill_with_neighbour_mean, start=0, n_steps=3, seq_length=4
        )
        torch.testing.assert_close(preds, torch.full((3, 5, 5), 12.5))
        self.assertEqual(masks.sum().item(), 3 * 9)

    def test_linear_ramp_preserved_on_interior(self):
        """The 3x3 mean of a linear field is the centre value."""
        sst = ramp_grid().expand(4, 5, 5).clone()
        preds, _, masks = rollout.rollout_neighbour_average(
            sst, rollout.fill_with_ground_truth, start=0, n_steps=2, seq_length=2
        )
        for k in range(2):
            torch.testing.assert_close(preds[k][masks[k]], ramp_grid()[masks[k]])

    def test_same_cells_and_truth_as_ca_rnn(self):
        sst = torch.stack([ramp_grid() + t for t in range(10)])
        sst[:, 0, 0] = NAN
        sst[:, 2, 3] = NAN
        base = rollout.rollout_neighbour_average(
            sst, rollout.fill_with_neighbour_mean, start=1, n_steps=3, seq_length=4
        )
        model = rollout.rollout_ca_rnn(
            Persistence(), sst, 0.0, 1.0, rollout.fill_with_neighbour_mean,
            start=1, n_steps=3, seq_length=4,
        )
        _, truth_baseline, masks_baseline = base
        _, truth_model, masks_model = model
        torch.testing.assert_close(truth_baseline, truth_model, equal_nan=True)
        self.assertTrue(torch.equal(masks_baseline, masks_model))


class TestRolloutPersistence(unittest.TestCase):
    def test_module_returns_centre_of_last_frame(self):
        x = torch.arange(2 * 3 * 9, dtype=torch.float32).view(2, 3, 9)  # (n_cells, seq_length, 9)
        torch.testing.assert_close(rollout.Persistence()(x), x[:, -1, 4:5])

    def test_repeats_last_seed_frame_on_predicted_cells(self):
        sst = torch.stack([ramp_grid() + t for t in range(8)])
        preds, truth, masks = rollout.rollout_persistence(
            sst, rollout.fill_with_ground_truth, start=1, n_steps=3, seq_length=4
        )
        last_seed = sst[4]
        for k in range(3):
            torch.testing.assert_close(preds[k][masks[k]], last_seed[masks[k]])
        torch.testing.assert_close(truth, sst[5:8])

    def test_same_cells_as_neighbour_average(self):
        sst = torch.stack([ramp_grid() + t for t in range(10)])
        sst[:, 0, 0] = NAN
        sst[:, 2, 3] = NAN
        _, truth_p, masks_p = rollout.rollout_persistence(
            sst, rollout.fill_with_neighbour_mean, start=1, n_steps=3, seq_length=4
        )
        _, truth_a, masks_a = rollout.rollout_neighbour_average(
            sst, rollout.fill_with_neighbour_mean, start=1, n_steps=3, seq_length=4
        )
        torch.testing.assert_close(truth_p, truth_a, equal_nan=True)
        self.assertTrue(torch.equal(masks_p, masks_a))


class TestErrorsPerStep(unittest.TestCase):
    def test_masked_mae_rmse(self):
        truth = torch.zeros(2, 2, 2)
        preds = torch.tensor([
            [[1.0, -3.0], [100.0, NAN]],  # masked cells: 1, -3
            [[2.0, 2.0], [2.0, 2.0]],     # masked cells: all four
        ])
        masks = torch.tensor([
            [[True, True], [False, False]],
            [[True, True], [True, True]],
        ])
        errors = rollout.errors_per_step(preds, truth, masks)
        self.assertEqual(set(errors), {"mae", "rmse"})
        self.assertAlmostEqual(errors["mae"][0], 2.0)            # (1 + 3) / 2
        self.assertAlmostEqual(errors["rmse"][0], math.sqrt(5))  # sqrt((1 + 9) / 2)
        self.assertAlmostEqual(errors["mae"][1], 2.0)
        self.assertAlmostEqual(errors["rmse"][1], 2.0)


if __name__ == "__main__":
    unittest.main()
