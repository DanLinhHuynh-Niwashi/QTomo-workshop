"""Records of a tomography run, persisted as NumPy ``.npz`` files that ``np.load`` reads directly.

Typical use::

    from qtomo_skeleton.writer import Evaluation, TomographyResult

    evaluation = Evaluation.from_model(tomo, test_states, target_process_fn)
    result = TomographyResult.from_model(tomo, evaluation, gamma=0.1, seed=0)
    path = result.to_npz("output/my_experiment/run0")
    same = TomographyResult.from_npz(path)
"""

from qtomo_skeleton.writer.result import FORMAT_VERSION, Evaluation, TomographyResult, get_value, group_by

__all__ = [
    "FORMAT_VERSION",
    "Evaluation",
    "TomographyResult",
    "get_value",
    "group_by",
]
