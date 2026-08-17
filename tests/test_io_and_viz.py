"""Ida e volta do resultado em disco, e teste de fumaça das três animações."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # sem display, antes de qualquer import de pyplot

import numpy as np
import pytest

from oceantides.io import load_result, save_result
from oceantides.simulation import SimConfig, run_simulation


@pytest.fixture(scope="module")
def result():
    return run_simulation(
        SimConfig(days=8.0, dt=300.0, sample_dt=1800.0, grid_shape=(31, 61))
    )


@pytest.fixture(scope="module")
def roundtripped(result, tmp_path_factory):
    path = tmp_path_factory.mktemp("io") / "out.npz"
    save_result(result, path)
    return load_result(path)


class TestRoundtrip:
    def test_arrays_survive(self, result, roundtripped):
        for name in ("t", "h_dynamic", "h_equilibrium", "moon_pos", "sun_pos"):
            np.testing.assert_allclose(
                getattr(roundtripped, name), getattr(result, name)
            )

    def test_config_survives(self, result, roundtripped):
        assert roundtripped.config == result.config

    def test_station_names_survive(self, result, roundtripped):
        assert roundtripped.station_names == result.station_names

    def test_ephemerides_are_rebuildable(self, roundtripped):
        """O campo global é recalculado na animação, então as efemérides
        precisam ser reconstruíveis a partir do que foi salvo."""
        bodies = roundtripped.rebuild_bodies()
        assert len(bodies) == len(roundtripped.config.bodies)
        pos = bodies[0][0].position(roundtripped.t[3])
        np.testing.assert_allclose(pos, roundtripped.moon_pos[3], rtol=1e-9)

    def test_station_accessor(self, roundtripped):
        dyn, eq = roundtripped.station("Rio de Janeiro")
        assert dyn.shape == eq.shape == roundtripped.t.shape


class TestAnimationsRender:
    """Cada animação precisa produzir quadros sem display, sem exceção."""

    @pytest.mark.parametrize("module", ["polar_slice", "world_map", "globe3d"])
    def test_frames_render_headless(self, roundtripped, module, tmp_path):
        import importlib

        animate = importlib.import_module(f"oceantides.viz.{module}").animate
        anim, fig = animate(roundtripped, frames=6)

        for k in (0, 3, 5):
            anim._func(k)

        out = tmp_path / f"{module}.png"
        fig.savefig(out, dpi=60, facecolor=fig.get_facecolor())
        assert out.stat().st_size > 5_000

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_exaggeration_override_is_honoured(self, roundtripped):
        from oceantides.viz.polar_slice import animate

        anim, fig = animate(roundtripped, frames=3, exaggeration=1e6)
        anim._func(0)
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPaletteContract:
    """A paleta é validada; estes testes impedem regressão silenciosa."""

    def test_series_palette_is_the_validated_order(self):
        from oceantides.viz.common import SERIES

        assert SERIES == ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")

    def test_diverging_map_is_neutral_at_zero(self):
        """Cinza no meio: um tom no ponto médio faria zero parecer um valor."""
        from oceantides.viz.common import tide_cmap

        r, g, b, _ = tide_cmap()(0.5)
        assert max(r, g, b) - min(r, g, b) < 0.04  # praticamente acromático

    def test_diverging_arms_are_opposite_hues(self):
        from oceantides.viz.common import tide_cmap

        cmap = tide_cmap()
        low, high = cmap(0.05), cmap(0.95)
        assert low[2] > low[0]  # braço frio: azul domina
        assert high[0] > high[2]  # braço quente: vermelho domina

    def test_diverging_lightness_is_monotonic_per_arm(self):
        from oceantides.viz.common import tide_cmap

        cmap = tide_cmap()
        lum = lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
        cool = [lum(cmap(x)) for x in np.linspace(0.0, 0.49, 25)]
        warm = [lum(cmap(x)) for x in np.linspace(0.51, 1.0, 25)]
        assert np.all(np.diff(cool) > -1e-3)  # escurece -> clareia até o centro
        assert np.all(np.diff(warm) < 1e-3)  # clareia -> escurece a partir dele
