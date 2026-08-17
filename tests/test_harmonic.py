"""Constituintes harmônicas: frequências, números de Doodson e fatores nodais."""

from __future__ import annotations

import numpy as np
import pytest

from oceantides.harmonic import (
    CONSTITUENTS,
    FUNDAMENTAL,
    MAJOR_EIGHT,
    astronomical_arguments,
    harmonic_fit,
    nodal_factors,
    predict,
)

# Velocidades de referência em graus/hora (NOAA CO-OPS / Schureman)
REFERENCE_SPEEDS = {
    "M2": 28.9841042, "S2": 30.0000000, "N2": 28.4397295, "K2": 30.0821373,
    "K1": 15.0410686, "O1": 13.9430356, "P1": 14.9589314, "Q1": 13.3986609,
    "Mf": 1.0980331, "Mm": 0.5443747, "Ssa": 0.0821373, "Sa": 0.0410686,
    "M4": 57.9682084, "MS4": 58.9841042, "MN4": 57.4238337, "M6": 86.9523127,
    "L2": 29.5284789, "T2": 29.9589333, "NU2": 28.5125831, "MU2": 27.9682084,
    "2N2": 27.8953548, "J1": 15.5854433, "OO1": 16.1391017, "M1": 14.4966939,
    "MSf": 1.0158958, "2SM2": 31.0158958,
}

# Números de Doodson publicados, na codificação "coeficiente + 5"
REFERENCE_DOODSON = {
    "M2": "255.555", "S2": "273.555", "N2": "245.655", "K2": "275.555",
    "K1": "165.555", "O1": "145.555", "P1": "163.555", "Q1": "135.655",
    "Mf": "075.555", "Mm": "065.455", "Ssa": "057.555",
    "M4": "455.555", "MS4": "473.555", "M6": "655.555",
}


class TestSpeeds:
    @pytest.mark.parametrize("name,speed", sorted(REFERENCE_SPEEDS.items()))
    def test_speed_matches_reference(self, name, speed):
        assert CONSTITUENTS[name].speed == pytest.approx(speed, abs=2e-7)

    def test_m2_period_is_the_lunar_semidiurnal(self):
        """12.4206 h -- o mesmo valor que test_periods obtém por FFT."""
        assert CONSTITUENTS["M2"].period_hours == pytest.approx(12.4206, abs=1e-4)

    def test_s2_period_is_exactly_twelve_hours(self):
        assert CONSTITUENTS["S2"].period == pytest.approx(43200.0, rel=1e-12)

    def test_solar_perigee_rate_is_not_a_typo(self):
        """O perigeu solar avança 1.7195 graus/século = 1.96e-6 graus/h.

        Escrever 2e-7 em vez de 2e-6 desloca T2 em 1.7e-6 graus/h. Erro
        pequeno, mas suficiente para T2 divergir da tabela publicada.
        """
        assert FUNDAMENTAL[5] == pytest.approx(1.7195 / (36525 * 24), rel=0.02)


class TestDoodsonNumbers:
    @pytest.mark.parametrize("name,code", sorted(REFERENCE_DOODSON.items()))
    def test_encoding_matches_published(self, name, code):
        d = CONSTITUENTS[name].doodson
        expected = [int(c) for c in code.replace(".", "")]
        assert expected[0] == d[0]
        assert expected[1:] == [c + 5 for c in d[1:]]

    def test_species_is_the_first_doodson_number(self):
        for c in CONSTITUENTS.values():
            assert c.species == c.doodson[0]

    def test_species_partition(self):
        by_species = {}
        for c in CONSTITUENTS.values():
            by_species.setdefault(c.species, []).append(c.name)
        assert set(by_species[2]) >= {"M2", "S2", "N2", "K2"}
        assert set(by_species[1]) >= {"K1", "O1", "P1", "Q1"}
        assert set(by_species[0]) >= {"Mf", "Mm", "Ssa"}
        assert set(by_species[4]) >= {"M4", "MS4"}

    def test_shallow_water_have_no_equilibrium_amplitude(self):
        """M4, M6 e companhia NÃO existem no potencial astronômico.

        Nascem da não-linearidade local (atrito quadrático, continuidade) em
        água rasa -- é por isso que só um modelo dinâmico as produz, e a teoria
        de equilíbrio jamais poderia.
        """
        for name in ("M4", "MS4", "MN4", "M6", "2SM2"):
            assert CONSTITUENTS[name].amplitude == 0.0

    def test_major_eight_are_the_biggest_astronomical_ones(self):
        astronomical = [c for c in CONSTITUENTS.values()
                        if c.amplitude > 0 and c.species in (1, 2)]
        top = sorted(astronomical, key=lambda c: -c.amplitude)[:8]
        assert {c.name for c in top} == set(MAJOR_EIGHT)


class TestAmplitudeRatios:
    def test_s2_over_m2_is_the_solar_lunar_ratio(self):
        """0.46 -- o mesmo número que o PDF cita na p. 202."""
        ratio = CONSTITUENTS["S2"].amplitude / CONSTITUENTS["M2"].amplitude
        assert ratio == pytest.approx(0.46, abs=0.01)

    def test_m2_is_the_largest_constituent(self):
        assert max(CONSTITUENTS.values(), key=lambda c: c.amplitude).name == "M2"

    def test_k1_is_the_largest_diurnal(self):
        diurnal = [c for c in CONSTITUENTS.values() if c.species == 1]
        assert max(diurnal, key=lambda c: c.amplitude).name == "K1"


class TestNodalFactors:
    """Modulação de 18.6 anos. Sem ela, constantes ajustadas num ano não
    valem no seguinte."""

    @staticmethod
    def _range(name):
        days = np.linspace(0.0, 7000.0, 4000)  # ~19 anos, cobre um ciclo nodal
        f, _ = nodal_factors(name, days)
        f = np.atleast_1d(f)
        return float(f.min()), float(f.max())

    def test_m2_varies_by_about_four_percent(self):
        lo, hi = self._range("M2")
        assert (hi - lo) / 2 == pytest.approx(0.037, abs=0.005)

    def test_k2_is_the_most_sensitive(self):
        lo, hi = self._range("K2")
        assert (hi - lo) / 2 == pytest.approx(0.286, abs=0.02)

    def test_o1_varies_by_about_eighteen_percent(self):
        lo, hi = self._range("O1")
        assert (hi - lo) / 2 == pytest.approx(0.187, abs=0.02)

    def test_solar_constituents_have_no_nodal_modulation(self):
        """S2 e P1 são puramente solares: o nó lunar não os afeta."""
        for name in ("S2", "P1"):
            lo, hi = self._range(name)
            assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)

    def test_ordering_of_sensitivity(self):
        spread = lambda n: self._range(n)[1] - self._range(n)[0]
        assert spread("K2") > spread("O1") > spread("K1") > spread("M2")

    def test_shallow_water_inherits_from_parent(self):
        """M4 é a sobremaré de M2, então seu fator nodal é f(M2) ao quadrado."""
        days = np.array([1234.0])
        f_m2, _ = nodal_factors("M2", days)
        f_m4, _ = nodal_factors("M4", days)
        np.testing.assert_allclose(f_m4, np.asarray(f_m2) ** 2)


class TestAstronomicalArguments:
    def test_longitudes_at_j2000(self):
        s, h, p, n, ps = astronomical_arguments(0.0)
        assert float(s) == pytest.approx(218.316, abs=0.01)
        assert float(h) == pytest.approx(280.466, abs=0.01)
        assert float(n) == pytest.approx(125.045, abs=0.01)

    def test_all_within_zero_and_360(self):
        days = np.linspace(-40000, 40000, 500)
        for arg in astronomical_arguments(days):
            assert np.all((arg >= 0) & (arg < 360))

    def test_moon_longitude_advances_one_revolution_per_sidereal_month(self):
        s0 = astronomical_arguments(0.0)[0]
        s1 = astronomical_arguments(27.321661)[0]
        assert float(s1) == pytest.approx(float(s0), abs=0.05)

    def test_node_regresses_over_18_6_years(self):
        n0 = astronomical_arguments(0.0)[3]
        n1 = astronomical_arguments(18.612958 * 365.25)[3]
        assert float(n1) == pytest.approx(float(n0), abs=0.3)


class TestHarmonicFit:
    """O método de Darwin: frequências da astronomia, amplitudes dos dados."""

    def test_recovers_a_synthetic_tide(self):
        names = ["M2", "S2", "N2", "K1", "O1"]
        truth = {
            "M2": (1.30, 45.0), "S2": (0.40, 90.0), "N2": (0.25, 12.0),
            "K1": (0.18, 200.0), "O1": (0.11, 310.0),
        }
        t = np.arange(0, 60 * 86400, 600.0)
        series = predict(t, truth, mean=0.7)

        fit = harmonic_fit(t, series, names)
        assert fit["mean"] == pytest.approx(0.7, abs=1e-6)
        for name, (amp, phase) in truth.items():
            assert fit[name][0] == pytest.approx(amp, rel=1e-4)
            assert fit[name][1] == pytest.approx(phase, abs=0.05)

    def test_roundtrip_reconstructs_the_series(self):
        names = ["M2", "S2", "K1"]
        t = np.arange(0, 40 * 86400, 900.0)
        truth = {"M2": (1.0, 30.0), "S2": (0.3, 120.0), "K1": (0.2, 260.0)}
        series = predict(t, truth)
        fit = harmonic_fit(t, series, names, include_mean=False)
        np.testing.assert_allclose(predict(t, fit), series, atol=1e-8)

    def test_rayleigh_criterion_amplifies_noise(self):
        """Separar S2 de K2 exige ~183 dias: ``1/(30.0821 - 30.0000)`` graus/h.

        Nuance que é fácil errar: com dados sintéticos **sem ruído** contendo
        exatamente as duas constituintes, mínimos quadrados as separa mesmo num
        registro curto -- as colunas do sistema não chegam a ser colineares.

        O critério de Rayleigh não é sobre singularidade, é sobre
        **condicionamento**: num registro curto as duas colunas ficam quase
        paralelas, o número de condição explode, e qualquer ruído (ou sinal não
        modelado) é amplificado na mesma proporção. Este teste mede exatamente
        isso.
        """
        truth = {"S2": (0.40, 0.0), "K2": (0.11, 0.0)}

        def condition_and_error(days, step, trials=200):
            t = np.arange(0, days * 86400, step)
            omega = [CONSTITUENTS[n].omega for n in ("S2", "K2")]
            design = np.stack(
                [f(w * t) for w in omega for f in (np.cos, np.sin)], axis=-1
            )
            base = predict(t, truth)
            rng = np.random.default_rng(42)  # mesma semente por caso
            errs = [
                abs(
                    harmonic_fit(
                        t, base + rng.normal(0.0, 0.01, t.size), ["S2", "K2"],
                        include_mean=False,
                    )["K2"][0] - 0.11
                )
                for _ in range(trials)
            ]
            return np.linalg.cond(design), float(np.sqrt(np.mean(np.square(errs))))

        # período de Rayleigh de S2-K2: 360/((30.0821373 - 30) * 24) dias
        rayleigh_days = 360.0 / (
            (CONSTITUENTS["K2"].speed - CONSTITUENTS["S2"].speed) * 24.0
        )
        assert rayleigh_days == pytest.approx(182.6, abs=1.0)

        cond15, err15 = condition_and_error(15, 900.0)
        cond30, err30 = condition_and_error(30, 900.0)
        cond_ray, err_ray = condition_and_error(round(rayleigh_days), 1800.0)

        # o condicionamento melhora monotonicamente até o período de Rayleigh,
        # onde as duas colunas ficam exatamente ortogonais
        assert cond15 > cond30 > cond_ray
        assert cond_ray == pytest.approx(1.0, abs=0.05)

        # e o erro induzido por ruído segue o número de condição de perto:
        # é literalmente o fator de amplificação
        assert err15 / err_ray == pytest.approx(cond15, rel=0.5)
        assert err15 > 8 * err_ray

class TestAgainstNOAA:
    """Verdade independente: as previsões oficiais da NOAA.

    Se a máquina harmônica reconstrói as tábuas de maré da NOAA a partir das
    constantes que a própria NOAA publica, então argumentos astronômicos,
    fatores nodais e correções de fase estão todos corretos. É o teste mais
    forte deste módulo, e o que revelou que as correções de fase diurnas
    estavam com o sinal invertido (erro de exatamente 180 graus em K1, O1, P1
    e Q1, com as semidiurnas já exatas).
    """

    @staticmethod
    def _data(station="1612340", begin="20260301", end="20260331"):
        pytest.importorskip("urllib.request")
        from oceantides.validation import fetch_predictions, load_reference

        try:
            days, obs = fetch_predictions(station, begin, end)
            constants = load_reference(verbose=False)[station]
        except Exception as exc:  # rede indisponível
            pytest.skip(f"API da NOAA inacessível: {exc}")
        return days, obs, constants

    @pytest.mark.parametrize(
        "station,name,tolerance",
        [
            ("1612340", "Honolulu", 0.07),
            ("9414290", "San Francisco", 0.09),
            ("8443970", "Boston", 0.14),
            ("8724580", "Key West", 0.10),
        ],
    )
    def test_reconstructs_official_predictions(self, station, name, tolerance):
        from oceantides.harmonic import predict_from_constants

        days, obs, constants = self._data(station)
        mine = predict_from_constants(constants, days)

        assert np.corrcoef(mine, obs)[0, 1] > 0.98, name
        assert np.sqrt(np.mean((mine - obs) ** 2)) < tolerance, name

    def test_recovered_constants_match_the_published_ones(self):
        """Ajustando as previsões com os nossos argumentos, devemos recuperar
        exatamente as constantes publicadas -- amplitude e fase."""
        from oceantides.harmonic import astronomical_argument, nodal_factors

        # Um ano inteiro: com 3 meses, N2 ainda sofre vazamento de L2, nu2 e
        # 2N2, que não estão no ajuste, e sua fase desvia ~9 graus.
        days, obs, constants = self._data(end="20261231")
        names = list(MAJOR_EIGHT)

        cols = []
        for n in names:
            arg = np.radians(astronomical_argument(n, days))
            f, _ = nodal_factors(n, days)
            cols += [f * np.cos(arg), f * np.sin(arg)]
        design = np.stack(cols + [np.ones_like(days)], axis=-1)
        coeffs, *_ = np.linalg.lstsq(design, obs, rcond=None)

        for k, n in enumerate(names):
            a, b = coeffs[2 * k], coeffs[2 * k + 1]
            amp = float(np.hypot(a, b))
            phase = float(np.degrees(np.arctan2(b, a)) % 360.0)
            ref_amp, ref_phase = constants[n]

            assert amp == pytest.approx(ref_amp, abs=0.006), n
            delta = (phase - ref_phase + 180.0) % 360.0 - 180.0
            assert abs(delta) < 8.0, f"{n}: fase difere {delta:.1f} graus"

    def test_diurnal_phase_corrections_are_not_optional(self):
        """Sem elas a reconstrução desaba -- regressão do bug de 180 graus."""
        from oceantides import harmonic
        from oceantides.harmonic import predict_from_constants

        days, obs, constants = self._data()
        good = np.corrcoef(predict_from_constants(constants, days), obs)[0, 1]

        original = dict(harmonic.PHASE_CORRECTION)
        try:
            for name in ("K1", "O1", "P1", "Q1"):
                broken = harmonic.CONSTITUENTS[name]
                harmonic.CONSTITUENTS[name] = harmonic.Constituent(
                    broken.name, broken.doodson, broken.amplitude,
                    broken.description, -broken.phase_correction,
                )
            bad = np.corrcoef(predict_from_constants(constants, days), obs)[0, 1]
        finally:
            for name in ("K1", "O1", "P1", "Q1"):
                harmonic.CONSTITUENTS[name] = harmonic.Constituent(
                    harmonic.CONSTITUENTS[name].name,
                    harmonic.CONSTITUENTS[name].doodson,
                    harmonic.CONSTITUENTS[name].amplitude,
                    harmonic.CONSTITUENTS[name].description,
                    original[name],
                )
        assert good > 0.98 and bad < 0.8, (good, bad)


class TestMoreHarmonicFit:
    def test_eight_constituents_capture_most_of_the_signal(self):
        """Os 8 principais devem reter a maior parte da variância."""
        t = np.arange(0, 90 * 86400, 900.0)
        full = {n: (CONSTITUENTS[n].amplitude, 30.0 * k)
                for k, n in enumerate(REFERENCE_SPEEDS) if CONSTITUENTS[n].amplitude > 0}
        series = predict(t, full)
        fit8 = harmonic_fit(t, series, list(MAJOR_EIGHT), include_mean=False)
        residual = series - predict(t, fit8)
        assert residual.std() / series.std() < 0.35
