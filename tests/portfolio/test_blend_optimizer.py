"""Tests for generate_blend_grid in blend_optimizer."""

import pytest

from python.portfolio.blend_optimizer import generate_blend_grid


class TestGenerateBlendGridTwoMethods:
    """Two methods, step=0.2: valid combos summing to 1.0."""

    def test_all_blends_nonempty(self):
        blends = generate_blend_grid(["A", "B"], step=0.2)
        assert len(blends) > 0

    def test_expected_count(self):
        # step=0.2 with 2 methods: (0.0,1.0),(0.2,0.8),...,(1.0,0.0) = 6 combos
        # But zero-weight methods are excluded, so pure single-method entries
        # have only 1 key while mixed have 2. All 6 combos are valid.
        blends = generate_blend_grid(["A", "B"], step=0.2)
        assert len(blends) == 6

    def test_known_values(self):
        blends = generate_blend_grid(["A", "B"], step=0.2)
        blend_tuples = {tuple(sorted(b.items())) for b in blends}
        expected = {
            (("B", 1.0),),
            (("A", 0.2), ("B", 0.8)),
            (("A", 0.4), ("B", 0.6)),
            (("A", 0.6), ("B", 0.4)),
            (("A", 0.8), ("B", 0.2)),
            (("A", 1.0),),
        }
        assert blend_tuples == expected


class TestWeightsSumToOne:
    """Every blend's weights must sum to 1.0 within float tolerance."""

    @pytest.mark.parametrize(
        "methods,step",
        [
            (["X", "Y"], 0.2),
            (["A", "B", "C"], 0.5),
            (["A", "B", "C"], 0.2),
            (["P", "Q", "R", "S"], 0.5),
        ],
    )
    def test_weights_sum(self, methods, step):
        blends = generate_blend_grid(methods, step=step)
        for blend in blends:
            assert abs(sum(blend.values()) - 1.0) < 1e-6, (
                f"Blend {blend} sums to {sum(blend.values())}"
            )


class TestNoZeroWeights:
    """Zero-weight methods must be excluded from blend dicts."""

    def test_no_zeros_two_methods(self):
        blends = generate_blend_grid(["A", "B"], step=0.2)
        for blend in blends:
            for method, weight in blend.items():
                assert weight > 0, f"Zero weight found for {method} in {blend}"

    def test_no_zeros_three_methods(self):
        blends = generate_blend_grid(["A", "B", "C"], step=0.5)
        for blend in blends:
            for method, weight in blend.items():
                assert weight > 0, f"Zero weight found for {method} in {blend}"


class TestSingleMethod:
    """Single method: only one valid blend with weight 1.0."""

    def test_single_method_count(self):
        blends = generate_blend_grid(["only"], step=0.2)
        assert len(blends) == 1

    def test_single_method_value(self):
        blends = generate_blend_grid(["only"], step=0.2)
        assert blends[0] == {"only": 1.0}

    def test_single_method_step_half(self):
        blends = generate_blend_grid(["solo"], step=0.5)
        assert len(blends) == 1
        assert blends[0] == {"solo": 1.0}


class TestThreeMethodsStepHalf:
    """Three methods, step=0.5: known small set."""

    def test_exact_output(self):
        blends = generate_blend_grid(["A", "B", "C"], step=0.5)
        blend_tuples = {tuple(sorted(b.items())) for b in blends}
        expected = {
            (("A", 1.0),),
            (("B", 1.0),),
            (("C", 1.0),),
            (("A", 0.5), ("B", 0.5)),
            (("A", 0.5), ("C", 0.5)),
            (("B", 0.5), ("C", 0.5)),
        }
        assert blend_tuples == expected

    def test_count(self):
        blends = generate_blend_grid(["A", "B", "C"], step=0.5)
        assert len(blends) == 6


class TestStepOne:
    """Step=1.0: each method gets exactly one blend with weight=1.0."""

    def test_two_methods(self):
        blends = generate_blend_grid(["X", "Y"], step=1.0)
        assert len(blends) == 2
        blend_tuples = {tuple(sorted(b.items())) for b in blends}
        assert blend_tuples == {(("X", 1.0),), (("Y", 1.0),)}

    def test_three_methods(self):
        blends = generate_blend_grid(["A", "B", "C"], step=1.0)
        assert len(blends) == 3
        blend_tuples = {tuple(sorted(b.items())) for b in blends}
        assert blend_tuples == {(("A", 1.0),), (("B", 1.0),), (("C", 1.0),)}


class TestFinerStep:
    """Step=0.1: more blends, all valid."""

    def test_two_methods_count(self):
        blends = generate_blend_grid(["A", "B"], step=0.1)
        # (0.0,1.0),(0.1,0.9),...,(1.0,0.0) = 11 combos
        assert len(blends) == 11

    def test_two_methods_all_sum_to_one(self):
        blends = generate_blend_grid(["A", "B"], step=0.1)
        for blend in blends:
            assert abs(sum(blend.values()) - 1.0) < 1e-6

    def test_three_methods_all_valid(self):
        blends = generate_blend_grid(["A", "B", "C"], step=0.1)
        assert len(blends) > 0
        for blend in blends:
            assert abs(sum(blend.values()) - 1.0) < 1e-6
            for w in blend.values():
                assert w > 0

    def test_three_methods_more_than_coarser(self):
        coarse = generate_blend_grid(["A", "B", "C"], step=0.5)
        fine = generate_blend_grid(["A", "B", "C"], step=0.1)
        assert len(fine) > len(coarse)
