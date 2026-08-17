"""Os períodos de maré emergem da astronomia, não são inseridos à mão.

Nenhuma frequência de maré aparece no código: só as constantes orbitais. Estes
testes verificam que M2, S2 e o batimento de sizígia/quadratura *saem* da
simulação, confirmando toda a cadeia efeméride -> rotação -> forçamento.
"""

from __future__ import annotations

import numpy as np
import pytest

from oceantides.constants import (
    T_LUNAR_DAY,
    T_M2,
    T_S2,
    T_SPRING_NEAP,
)
from oceantides.simulation import SimConfig, build_bodies, make_forcing
from oceantides.stations import get_stations, station_positions

SAMPLE_DT = 900.0
DAYS = 120.0


@pytest.fixture(scope="module")
def series():
    """Maré de equilíbrio em Rio de Janeiro, amostrada por 120 dias.

    Usa a maré de equilíbrio (forma fechada) e não a resposta dinâmica: os
    períodos vêm da astronomia, e assim o teste não fica refém do transiente
    da bacia nem dos parâmetros ajustados.
    """
    cfg = SimConfig(ephemeris_mode="kepler")
    forcing = make_forcing(
        station_positions(get_stations(["Rio de Janeiro"])), build_bodies(cfg)
    )
    t = np.arange(0.0, DAYS * 86400.0, SAMPLE_DT)
    h = np.array([forcing(ti)[0] for ti in t])
    return t, h


def _spectrum(t, h):
    """Periodograma com janela de Hann, devolvendo (períodos [s], potência)."""
    x = h - h.mean()
    x = x * np.hanning(x.size)
    power = np.abs(np.fft.rfft(x)) ** 2
    freq = np.fft.rfftfreq(x.size, d=t[1] - t[0])
    with np.errstate(divide="ignore"):
        periods = np.where(freq > 0, 1.0 / np.where(freq > 0, freq, 1), np.inf)
    return periods, power


def _peak_period(periods, power, lo, hi):
    band = (periods >= lo) & (periods <= hi)
    idx = np.where(band)[0]
    return periods[idx[np.argmax(power[idx])]]


class TestSemidiurnalLines:
    def test_m2_is_12h25min(self, series):
        """A componente lunar semidiurna. O PDF arredonda para 12h26min."""
        periods, power = _spectrum(*series)
        peak = _peak_period(periods, power, 12.2 * 3600, 12.7 * 3600)
        assert peak == pytest.approx(T_M2, rel=0.005)
        assert peak / 3600 == pytest.approx(12.42, abs=0.06)

    def test_s2_is_exactly_12h(self, series):
        """A componente solar semidiurna: metade de um dia solar, por definição."""
        periods, power = _spectrum(*series)
        peak = _peak_period(periods, power, 11.85 * 3600, 12.15 * 3600)
        assert peak == pytest.approx(T_S2, rel=0.005)

    def test_m2_dominates_s2(self, series):
        """A maré lunar é maior que a solar -- razão ~0.46 em amplitude."""
        periods, power = _spectrum(*series)
        m2 = power[np.argmin(np.abs(periods - T_M2))]
        s2 = power[np.argmin(np.abs(periods - T_S2))]
        assert np.sqrt(s2 / m2) == pytest.approx(0.46, abs=0.12)


class TestLunarDay:
    def test_two_high_tides_per_lunar_day(self, series):
        """A afirmação central do PDF (p. 202): duas marés altas por dia.

        E não por dia *solar*: a Lua avança na órbita, então o ciclo é o dia
        lunar de 24h50min -- 'they occur once every 12 h and 26 min' (p. 204).
        """
        t, h = series
        interior = h[1:-1]
        is_peak = (interior > h[:-2]) & (interior > h[2:])
        n_peaks = int(is_peak.sum())

        expected = 2.0 * (DAYS * 86400.0) / T_LUNAR_DAY
        assert n_peaks == pytest.approx(expected, rel=0.05)

    def test_lunar_day_is_longer_than_solar_day(self):
        assert T_LUNAR_DAY > 86400.0
        assert T_LUNAR_DAY / 3600 == pytest.approx(24.84, abs=0.02)


class TestSpringNeapBeat:
    def test_beat_period_is_14_8_days(self, series):
        """M2 e S2 batem no período sinódico/2 = 14.77 dias.

        Medido pelo espaçamento entre máximos da amplitude diária -- é o
        envelope que um marégrafo real mostra.
        """
        t, h = series
        per_day = int(round(86400.0 / SAMPLE_DT))
        n_days = h.size // per_day
        daily_range = np.ptp(h[: n_days * per_day].reshape(n_days, per_day), axis=1)

        # suaviza com 3 dias para remover a modulação de declinação
        kernel = np.ones(3) / 3.0
        smooth = np.convolve(daily_range, kernel, mode="valid")
        interior = smooth[1:-1]
        peaks = np.where((interior > smooth[:-2]) & (interior > smooth[2:]))[0] + 1

        assert len(peaks) >= 5, f"esperado ~8 sizígias em {DAYS:.0f} dias, achei {len(peaks)}"
        spacing = np.diff(peaks).mean()  # em dias
        assert spacing == pytest.approx(T_SPRING_NEAP / 86400.0, rel=0.10)

    def test_spring_exceeds_neap(self, series):
        t, h = series
        per_day = int(round(86400.0 / SAMPLE_DT))
        n_days = h.size // per_day
        daily = np.ptp(h[: n_days * per_day].reshape(n_days, per_day), axis=1)
        assert daily.max() / daily.min() > 2.0

    def test_synodic_month_from_constants(self):
        """O batimento é metade do mês sinódico, não do sideral."""
        assert T_SPRING_NEAP / 86400.0 == pytest.approx(14.765, abs=0.01)


class TestDiurnalInequality:
    """p. 204: 'one high tide each day to be slightly higher than the other'."""

    def test_consecutive_high_tides_differ(self, series):
        t, h = series
        interior = h[1:-1]
        peak_idx = np.where((interior > h[:-2]) & (interior > h[2:]))[0] + 1
        heights = h[peak_idx]
        # a diferença entre marés altas consecutivas é sistematicamente não nula
        alternating = np.abs(np.diff(heights))
        assert alternating.mean() > 0.02 * np.ptp(h)

    def test_vanishes_at_zero_declination(self):
        """Sem inclinação orbital, as duas marés altas do dia são iguais.

        Confirma que a desigualdade vem mesmo da declinação, e não de outro
        efeito: o modo 'circular' do PDF zera a inclinação e a assimetria some.
        """
        cfg = SimConfig(ephemeris_mode="circular", bodies=("moon",))
        forcing = make_forcing(
            station_positions(get_stations(["Rio de Janeiro"])), build_bodies(cfg)
        )
        t = np.arange(0.0, 10 * 86400.0, 600.0)
        h = np.array([forcing(ti)[0] for ti in t])

        interior = h[1:-1]
        peak_idx = np.where((interior > h[:-2]) & (interior > h[2:]))[0] + 1
        heights = h[peak_idx][2:-2]
        assert np.ptp(heights) < 0.02 * np.ptp(h)
