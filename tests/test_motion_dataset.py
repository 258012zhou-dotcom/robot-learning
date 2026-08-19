import pytest
import torch

from robot_learning.motion_dataset import (
    StandardizationStats,
    fit_input_standardization,
    generate_motion_samples,
    split_motion_dataset,
    standardize_motion_inputs,
)


def test_generated_motion_samples_have_expected_shapes_and_targets():
    inputs, targets = generate_motion_samples(
        sample_count=20,
        dt=0.1,
        seed=42,
    )

    assert inputs.shape == (20, 4)
    assert targets.shape == (20, 2)
    torch.testing.assert_close(
        targets,
        inputs[:, :2] + inputs[:, 2:] * 0.1,
    )


def test_motion_sample_generation_is_reproducible():
    first = generate_motion_samples(10, dt=0.1, seed=42)
    second = generate_motion_samples(10, dt=0.1, seed=42)

    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])


def test_different_seed_changes_generated_inputs():
    first_inputs, _ = generate_motion_samples(10, dt=0.1, seed=1)
    second_inputs, _ = generate_motion_samples(10, dt=0.1, seed=2)

    assert not torch.equal(first_inputs, second_inputs)


@pytest.mark.parametrize(
    ("sample_count", "dt", "seed"),
    [
        (0, 0.1, 42),
        (-1, 0.1, 42),
        (10, 0.0, 42),
        (10, -0.1, 42),
        (10, 0.1, True),
    ],
)
def test_generation_rejects_invalid_parameters(sample_count, dt, seed):
    with pytest.raises(ValueError):
        generate_motion_samples(sample_count, dt, seed)


def test_split_has_expected_sizes_and_no_overlapping_indices():
    inputs, targets = generate_motion_samples(100, dt=0.1, seed=42)

    splits = split_motion_dataset(
        inputs,
        targets,
        train_fraction=0.7,
        validation_fraction=0.15,
        seed=7,
    )

    assert splits.train_inputs.shape == (70, 4)
    assert splits.validation_inputs.shape == (15, 4)
    assert splits.test_inputs.shape == (15, 4)

    train_indices = set(splits.train_indices.tolist())
    validation_indices = set(splits.validation_indices.tolist())
    test_indices = set(splits.test_indices.tolist())

    assert train_indices.isdisjoint(validation_indices)
    assert train_indices.isdisjoint(test_indices)
    assert validation_indices.isdisjoint(test_indices)
    assert train_indices | validation_indices | test_indices == set(range(100))


def test_split_is_reproducible():
    inputs, targets = generate_motion_samples(20, dt=0.1, seed=42)

    first = split_motion_dataset(inputs, targets, 0.7, 0.15, seed=5)
    second = split_motion_dataset(inputs, targets, 0.7, 0.15, seed=5)

    assert torch.equal(first.train_indices, second.train_indices)
    assert torch.equal(first.validation_indices, second.validation_indices)
    assert torch.equal(first.test_indices, second.test_indices)


@pytest.mark.parametrize(
    ("train_fraction", "validation_fraction"),
    [
        (0.0, 0.2),
        (1.0, 0.2),
        (0.7, 0.0),
        (0.7, 1.0),
        (0.8, 0.2),
    ],
)
def test_split_rejects_invalid_fractions(
    train_fraction,
    validation_fraction,
):
    inputs, targets = generate_motion_samples(20, dt=0.1, seed=42)

    with pytest.raises(ValueError):
        split_motion_dataset(
            inputs,
            targets,
            train_fraction,
            validation_fraction,
            seed=42,
        )


def test_training_inputs_are_standardized_with_training_statistics():
    inputs, targets = generate_motion_samples(100, dt=0.1, seed=42)
    splits = split_motion_dataset(inputs, targets, 0.7, 0.15, seed=7)

    stats = fit_input_standardization(splits.train_inputs)
    standardized = standardize_motion_inputs(splits.train_inputs, stats)

    torch.testing.assert_close(
        standardized.mean(dim=0),
        torch.zeros(4),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        standardized.std(dim=0, correction=0),
        torch.ones(4),
        atol=1e-6,
        rtol=0,
    )


def test_validation_inputs_use_training_statistics():
    train_inputs = torch.tensor(
        [
            [0.0, 10.0, -2.0, 2.0],
            [2.0, 14.0, 2.0, 6.0],
        ]
    )
    validation_inputs = torch.tensor([[3.0, 18.0, 4.0, 8.0]])

    stats = fit_input_standardization(train_inputs)
    standardized = standardize_motion_inputs(validation_inputs, stats)

    torch.testing.assert_close(
        standardized,
        torch.tensor([[2.0, 3.0, 2.0, 2.0]]),
    )


def test_fitting_standardization_does_not_modify_inputs():
    train_inputs, _ = generate_motion_samples(20, dt=0.1, seed=42)
    original = train_inputs.clone()

    fit_input_standardization(train_inputs)

    torch.testing.assert_close(train_inputs, original)


def test_standardization_rejects_feature_without_variation():
    train_inputs = torch.ones(10, 4)

    with pytest.raises(ValueError, match="vary"):
        fit_input_standardization(train_inputs)


def test_standardization_rejects_invalid_statistics():
    inputs = torch.ones(2, 4)
    stats = StandardizationStats(
        mean=torch.zeros(4),
        standard_deviation=torch.tensor([1.0, 1.0, 0.0, 1.0]),
    )

    with pytest.raises(ValueError, match="positive"):
        standardize_motion_inputs(inputs, stats)
