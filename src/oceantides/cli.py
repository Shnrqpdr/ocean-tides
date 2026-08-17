"""Interface de linha de comando: ``oceantides run | bench | animate | stations``."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .simulation import SimConfig, run_simulation


def _add_sim_args(p):
    p.add_argument("--days", type=float, default=30.0, help="duração simulada [dias]")
    p.add_argument("--dt", type=float, default=60.0, help="passo do RK4 [s]")
    p.add_argument("--sample-dt", type=float, default=600.0, help="intervalo de saída [s]")
    p.add_argument(
        "--ephemeris",
        default="kepler",
        choices=["circular", "kepler", "nbody", "astropy"],
        help="'circular' reproduz a hipótese simplificada do PDF",
    )
    p.add_argument(
        "--bodies",
        default="moon,sun",
        help="corpos perturbadores, separados por vírgula (moon,sun)",
    )
    p.add_argument("--stations", default=None, help="nomes separados por vírgula")
    p.add_argument("--love", action="store_true", help="aplica o fator de Terra elástica")


def _config_from_args(a) -> SimConfig:
    return SimConfig(
        days=a.days,
        dt=a.dt,
        sample_dt=a.sample_dt,
        ephemeris_mode=a.ephemeris,
        bodies=tuple(b.strip() for b in a.bodies.split(",") if b.strip()),
        stations=tuple(s.strip() for s in a.stations.split(",")) if a.stations else None,
        love=a.love,
    )


def cmd_run(args) -> int:
    from .io import save_result

    result = run_simulation(_config_from_args(args))
    path = save_result(result, args.out)

    print(f"Simulados {args.days:g} dias, {result.t.size} amostras -> {path}")
    print()
    print(f"{'estacao':<18}{'faixa[m]':>10}{'observada':>11}{'equilibrio':>12}"
          f"{'ampl M2':>9}{'atraso[h]':>11}")
    print("-" * 71)
    from .stations import get_stations

    for i, station in enumerate(get_stations(result.config.stations)):
        print(
            f"{station.name:<18}{np.ptp(result.h_dynamic[:, i]):>10.2f}"
            f"{station.observed_range:>11.2f}"
            f"{np.ptp(result.h_equilibrium[:, i]):>12.2f}"
            f"{result.meta['m2_amplification'][i]:>9.2f}"
            f"{result.meta['m2_lag_hours'][i]:>11.2f}"
        )
    return 0


def cmd_bench(args) -> int:
    from .bench import format_bench, run_bench

    rows = run_bench(dts=tuple(args.dts), years=args.years)
    print(format_bench(rows, years=args.years))
    return 0


def cmd_lte(args) -> int:
    """Resolve as Equações de Maré de Laplace no oceano global."""
    from .io import save_solution
    from .lte import LaplaceTidalModel, LTEConfig

    cfg = LTEConfig(
        resolution=args.resolution,
        days=args.days,
        spin_up_days=args.spin_up,
        forcing=args.forcing,
        constituent=args.constituent,
        internal_drag=args.internal_drag,
        drag=args.drag,
        sal_beta=args.sal,
    )
    model = LaplaceTidalModel(cfg)
    print(model.summary())
    print()

    names = [c.strip() for c in args.constituents.split(",")] if args.constituents \
        else [args.constituent]

    def progress(k, total, _state):
        print(f"  {100 * k / total:5.1f}%", end="\r", flush=True)

    solution = model.run(constituents=names, progress=progress)
    path = save_solution(solution, args.out)
    print(f"\nsolução gravada em {path}\n")

    for name in names:
        stats = solution.global_stats(name)
        amph = solution.find_amphidromes(name, 0.03)
        print(
            f"{name}: média {stats['amplitude_media']:.3f} m  ·  "
            f"máx {stats['amplitude_maxima']:.2f} m  ·  "
            f"oceano profundo {stats['amplitude_media_oceano_profundo']:.3f} m  ·  "
            f"{len(amph)} anfidromos"
        )
    return 0


def cmd_chart(args) -> int:
    from .io import load_solution
    from .viz.cotidal import plot_cotidal_chart

    solution = load_solution(args.input)
    fig, _, amph = plot_cotidal_chart(solution, args.constituent)

    stats = solution.amphidrome_rotation_stats(args.constituent, amplitude_threshold=0.03)
    print(f"{len(amph)} pontos anfidrômicos de {args.constituent}")
    for hemisphere, s in stats.items():
        if s["total"]:
            print(
                f"  hemisfério {hemisphere}: {s['total']:2d} pontos, "
                f"{100 * s['fracao_esperada']:.0f}% no sentido que Coriolis prevê"
            )

    _show_or_save(fig, args.save, args.dpi)
    return 0


def cmd_validate(args) -> int:
    from .io import load_solution
    from .validation import ALL_STATIONS, compare, format_comparison
    from .viz.cotidal import plot_validation

    solution = load_solution(args.input)
    rows = compare(solution, args.constituent, ALL_STATIONS)
    print(format_comparison(rows, args.constituent))

    if args.save or args.plot:
        fig, _ = plot_validation(rows, args.constituent)
        _show_or_save(fig, args.save, args.dpi)
    return 0


def _show_or_save(fig, path, dpi=140):
    import matplotlib.pyplot as plt

    if path:
        fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
        print(f"figura salva em {path}")
    else:
        plt.show()


def cmd_animate(args) -> int:
    from .io import load_result, load_solution

    # os modos de onda operam sobre a solução harmônica das LTE;
    # os demais, sobre a série temporal das estações
    if args.mode in ("wave", "currents"):
        solution = load_solution(args.input)
        from .viz import tide_wave

        animate = tide_wave.animate_currents if args.mode == "currents" \
            else tide_wave.animate
        anim, fig = animate(
            solution, constituent=args.constituent,
            frames=args.frames or 120, interval=args.interval,
        )
        if args.save:
            _save_animation(anim, args.save, args.fps, args.dpi)
            print(f"animação salva em {args.save}")
        else:
            import matplotlib.pyplot as plt

            plt.show()
        return 0

    result = load_result(args.input)

    if args.mode == "polar":
        from .viz.polar_slice import animate
    elif args.mode == "map":
        from .viz.world_map import animate
    else:
        from .viz.globe3d import animate

    anim, fig = animate(
        result,
        exaggeration=args.exaggeration,
        frames=args.frames,
        interval=args.interval,
    )

    if args.save:
        _save_animation(anim, args.save, args.fps, args.dpi)
        print(f"animação salva em {args.save}")
    else:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


def _save_animation(anim, path, fps, dpi=100):
    from matplotlib.animation import FFMpegWriter, PillowWriter

    if str(path).lower().endswith(".gif"):
        anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        return

    # CRF em vez de bitrate fixo: qualidade constante, tamanho adaptado à cena.
    # 'yuv420p' é exigido pelo QuickTime e por vários players -- sem ele o vídeo
    # sai válido mas não abre em boa parte dos lugares. Requer dimensões pares,
    # e as figuras deste projeto já têm.
    anim.save(
        path,
        writer=FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=["-crf", "20", "-pix_fmt", "yuv420p", "-preset", "medium"],
        ),
        dpi=dpi,
    )


def cmd_stations(_args) -> int:
    from .constants import T_M2
    from .response import steady_state_response
    from .stations import DEFAULT_STATIONS

    omega = 2 * np.pi / T_M2
    print(f"{'estacao':<18}{'lat':>8}{'lon':>9}{'T0[h]':>8}{'Q':>7}"
          f"{'ampl':>8}{'atraso[h]':>11}{'obs[m]':>9}")
    print("-" * 78)
    for s in DEFAULT_STATIONS:
        amp, lag = steady_state_response(2 * np.pi / s.natural_period, s.q_factor, omega)
        print(
            f"{s.name:<18}{s.lat:>8.2f}{s.lon:>9.2f}{s.natural_period_h:>8.2f}"
            f"{s.q_factor:>7.1f}{float(amp):>8.2f}"
            f"{float(lag) / (2 * np.pi) * T_M2 / 3600:>11.2f}{s.observed_range:>9.2f}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oceantides",
        description=(
            "Simulador de marés oceânicas a partir do formalismo newtoniano de "
            "Thornton & Marion, §5.5."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="executa a simulação e grava o resultado")
    _add_sim_args(p_run)
    p_run.add_argument("--out", default="out.npz", help="arquivo .npz de saída")
    p_run.set_defaults(func=cmd_run)

    p_bench = sub.add_parser("bench", help="compara os integradores")
    p_bench.add_argument("--years", type=float, default=10.0)
    p_bench.add_argument("--dts", type=float, nargs="+", default=[600.0, 1800.0, 3600.0])
    p_bench.set_defaults(func=cmd_bench)

    p_lte = sub.add_parser(
        "lte", help="resolve as Equações de Maré de Laplace no oceano global"
    )
    p_lte.add_argument("--resolution", type=float, default=1.0, help="graus")
    p_lte.add_argument("--days", type=float, default=12.5)
    p_lte.add_argument("--spin-up", type=float, default=10.0)
    p_lte.add_argument("--forcing", default="constituent",
                       choices=["constituent", "astronomical"])
    p_lte.add_argument("--constituent", default="M2")
    p_lte.add_argument("--constituents", default=None,
                       help="lista para extrair (só faz sentido com --forcing astronomical)")
    p_lte.add_argument("--internal-drag", type=float, default=1e-5,
                       help="arrasto linear de maré interna [1/s]")
    p_lte.add_argument("--drag", type=float, default=0.0025,
                       help="coeficiente de atrito de fundo quadrático")
    p_lte.add_argument("--sal", type=float, default=0.09,
                       help="beta da aproximação escalar de auto-atração e carga")
    p_lte.add_argument("--out", default="lte.npz")
    p_lte.set_defaults(func=cmd_lte)

    p_chart = sub.add_parser("chart", help="carta cotidal a partir de uma solução LTE")
    p_chart.add_argument("--input", default="lte.npz")
    p_chart.add_argument("--constituent", default="M2")
    p_chart.add_argument("--save", default=None)
    p_chart.add_argument("--dpi", type=int, default=140)
    p_chart.set_defaults(func=cmd_chart)

    p_val = sub.add_parser("validate", help="compara com marégrafos da NOAA")
    p_val.add_argument("--input", default="lte.npz")
    p_val.add_argument("--constituent", default="M2")
    p_val.add_argument("--plot", action="store_true")
    p_val.add_argument("--save", default=None)
    p_val.add_argument("--dpi", type=int, default=140)
    p_val.set_defaults(func=cmd_validate)

    p_anim = sub.add_parser("animate", help="anima um resultado")
    p_anim.add_argument("--input", default="out.npz")
    p_anim.add_argument(
        "--mode", default="polar",
        choices=["polar", "map", "globe", "wave", "currents"],
        help="'wave' e 'currents' operam sobre uma solução LTE (oceantides lte)",
    )
    p_anim.add_argument("--constituent", default="M2")
    p_anim.add_argument(
        "--exaggeration", type=float, default=None,
        help="fator de exagero do bojo; padrão escolhido automaticamente",
    )
    p_anim.add_argument("--frames", type=int, default=None)
    p_anim.add_argument("--interval", type=int, default=40, help="ms entre quadros")
    p_anim.add_argument("--save", default=None, help="grava .mp4 ou .gif em vez de exibir")
    p_anim.add_argument("--fps", type=int, default=25)
    p_anim.add_argument("--dpi", type=int, default=100, help="resolução ao gravar")
    p_anim.set_defaults(func=cmd_animate)

    p_st = sub.add_parser("stations", help="lista as estações e sua resposta em M2")
    p_st.set_defaults(func=cmd_stations)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
