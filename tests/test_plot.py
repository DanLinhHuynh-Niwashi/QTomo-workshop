import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from qtomo_skeleton import plot  # noqa: E402
from qtomo_skeleton.writer import Evaluation, TomographyResult  # noqa: E402


def fake_result(num_qubits=1, epochs=10, seed=0, loss_history=None, **metadata) -> TomographyResult:
    rng = np.random.default_rng(seed)
    dim = 2**num_qubits
    states = rng.normal(size=(3, dim, dim)) + 0j
    fidelities = rng.uniform(0.9, 1.0, size=3)
    if loss_history is None:
        loss_history = np.sort(rng.uniform(1e-3, 1, size=epochs))[::-1]
    return TomographyResult(
        model="kraus",
        num_qubits=num_qubits,
        loss_history=np.asarray(loss_history, dtype=float),
        training_states=states,
        initial_params=states[:1],
        params=states[:1],
        evaluation=Evaluation(states, states, states, fidelities),
        metadata=metadata,
    )


@pytest.fixture
def sweep():
    return [
        fake_result(n, seed=i, gamma=gamma, run=run)
        for i, (n, gamma, run) in enumerate(
            (n, gamma, run) for n in (1, 2) for gamma in (0.01, 0.1, 1.0) for run in (0, 1)
        )
    ]


def lines(fig: Figure, index: int = 0):
    return fig.axes[index].get_lines()


# ------------------------------------------------------- fidelity vs iteration


def test_fidelity_curves_grouped(sweep):
    fig = plot.plot_fidelity_curves(sweep)
    assert isinstance(fig, Figure)
    assert [line.get_label() for line in lines(fig)] == ["1 qubit(s)", "2 qubit(s)"]
    assert len(fig.axes[0].collections) == 2  # min..max bands
    assert fig.axes[0].get_ylabel() == "Fidelity"


def test_fidelity_is_one_minus_loss():
    result = fake_result(loss_history=[0.8, 0.3, 0.05])
    (line,) = lines(plot.plot_fidelity_curves([result], group=None))
    np.testing.assert_allclose(line.get_xdata(), [1, 2, 3])
    np.testing.assert_allclose(line.get_ydata(), [0.2, 0.7, 0.95])


def test_fidelity_curves_average_runs(sweep):
    fig = plot.plot_fidelity_curves(sweep)
    one_qubit = [r for r in sweep if r.num_qubits == 1]
    expected = np.mean([1 - r.loss_history for r in one_qubit], axis=0)
    np.testing.assert_allclose(lines(fig)[0].get_ydata(), expected)


def test_fidelity_curves_group_by_several_keys(sweep):
    fig = plot.plot_fidelity_curves(sweep, group=("num_qubits", "gamma"), label_fmt="{0}/{1}")
    assert len(lines(fig)) == 6
    assert lines(fig)[0].get_label() == "1/0.01"


def test_fidelity_curves_reject_mixed_lengths():
    with pytest.raises(ValueError):
        plot.plot_fidelity_curves([fake_result(epochs=5), fake_result(epochs=6)])


def test_fidelity_curves_reject_empty():
    with pytest.raises(ValueError):
        plot.plot_fidelity_curves([])


# ------------------------------------------------------------ metric vs x


def test_metric_vs_averages_duplicates(sweep):
    fig = plot.plot_metric_vs(sweep, "gamma", "final_loss", group="num_qubits", logx=True)
    one_qubit = lines(fig)[0]
    np.testing.assert_allclose(one_qubit.get_xdata(), [0.01, 0.1, 1.0])
    expected = np.mean([r.final_loss for r in sweep if r.num_qubits == 1 and r.metadata["gamma"] == 0.01])
    assert one_qubit.get_ydata()[0] == pytest.approx(expected)


def test_metric_vs_reduce_min(sweep):
    fig = plot.plot_metric_vs(sweep, "gamma", "final_loss", group="num_qubits", reduce="min")
    expected = min(r.final_loss for r in sweep if r.num_qubits == 1 and r.metadata["gamma"] == 0.01)
    assert lines(fig)[0].get_ydata()[0] == pytest.approx(expected)
    with pytest.raises(ValueError):
        plot.plot_metric_vs(sweep, "gamma", reduce="sum")


def test_metric_vs_log_axis_keeps_non_positive_values():
    results = [fake_result(loss_history=[0.5, -3e-9], gamma=0.1), fake_result(seed=1, gamma=1.0)]
    fig = plot.plot_metric_vs(results, "gamma", "final_loss")
    assert lines(fig)[0].get_ydata()[0] > 0  # -3e-9 drawn at the floor instead of disappearing


def test_metric_vs_draws_on_given_axes(sweep):
    fig = Figure()
    ax = fig.add_subplot()
    assert plot.plot_metric_vs(sweep, "gamma", "test_infidelity", group="num_qubits", ax=ax) is fig
    assert [line.get_label() for line in ax.get_lines()] == ["1", "2"]


def test_metric_vs_integer_x_axis_has_integer_ticks(sweep):
    fig = plot.plot_metric_vs(sweep, "num_qubits", "final_loss")
    ticks = fig.axes[0].get_xticks()
    assert np.all(ticks == np.round(ticks))


def test_figure_noise_sweep(sweep):
    fig = plot.figure_noise_sweep(sweep, reduce="min")
    assert len(fig.axes) == 2
    assert all(ax.get_xscale() == "log" for ax in fig.axes)


# ------------------------------------------------------------ density matrices


def test_density_matrices():
    rho = np.array([[0.6, 0.2 - 0.1j], [0.2 + 0.1j, 0.4]])
    fig = plot.plot_density_matrices({"a": rho, "b": rho.conj()})
    assert len(fig.axes) == 4
    assert fig.axes[0].get_xticklabels()[1].get_text() == r"$|1\rangle$"


def test_density_matrices_two_qubit_labels():
    fig = plot.plot_density_matrices({"a": np.eye(4) / 4})
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()][-1] == r"$|11\rangle$"


def test_density_matrices_reject_mixed_shapes():
    with pytest.raises(ValueError):
        plot.plot_density_matrices({"a": np.eye(2), "b": np.eye(4)})


# ------------------------------------------------------------ inset + saving


def test_curves_with_inset():
    t = np.linspace(0, 5, 50)
    fig = plot.plot_curves_with_inset(
        {"exact": (t, np.cos(t)), "learned": (t, np.cos(t) + 0.01)},
        inset_xlim=(0, 1),
        styles={"learned": {"linestyle": ":", "marker": "o"}},
    )
    (main,) = fig.axes
    (inset,) = main.child_axes
    assert len(main.get_lines()) == len(inset.get_lines()) == 2
    assert inset.get_xlim() == (0, 1)
    assert main.get_lines()[1].get_linestyle() == ":"


def test_save_figure(tmp_path, sweep):
    paths = plot.save_figure(plot.plot_fidelity_curves(sweep), tmp_path / "figs" / "fig2", formats=("png", "eps"))
    assert [p.name for p in paths] == ["fig2.png", "fig2.eps"]
    assert all(p.stat().st_size > 0 for p in paths)
