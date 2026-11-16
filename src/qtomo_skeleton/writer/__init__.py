"""Persist tomography runs as NumPy ``.npz`` files that ``np.load`` reads directly.

Typical use::

    from qtomo_skeleton.writer import Evaluation, ExperimentWriter, TomographyResult

    evaluation = Evaluation.from_model(tomo, test_states, target_process_fn)
    ExperimentWriter("output/my_experiment").save(
        TomographyResult.from_model(tomo, evaluation, gamma=0.1, seed=0)
    )
"""

from qtomo_skeleton.writer.npz import FORMAT_VERSION, ExperimentWriter, load_result, load_results, save_result
from qtomo_skeleton.writer.result import Evaluation, TomographyResult, get_value, group_by

__all__ = [
    "FORMAT_VERSION",
    "Evaluation",
    "ExperimentWriter",
    "TomographyResult",
    "get_value",
    "group_by",
    "load_result",
    "load_results",
    "save_result",
]
