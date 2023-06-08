import deq
import deqflow2
import pytest


@pytest.fixture(params=[True, False])
def mod_deq(request) -> deq.DEQBase:
    use_indexing = request.param
    mod_cls = deq.DEQIndexing if use_indexing else deq.DEQSliced
    mod = mod_cls(solver=deq.solvers.naive_solver, threshold=50)

    return mod


@pytest.fixture(params=["tiny", "medium", "large", "huge", "gigantic"])
def mod_deqflow(request, mod_deq) -> deqflow2.DEQFlow:
    variant = deqflow2.Variant(request.param)
    mod = deqflow2.DEQFlow(variant=variant, deq=mod_deq)

    return mod
