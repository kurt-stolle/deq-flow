import pytest
import deqflow2

@pytest.fixture()
@pytest.mark.parametrize("use_indexing", [True, False])
def mod_deq(use_indexing) -> deqflow2.DEQ:
    mod_cls = deqflow2.deq.DEQIndexing if use_indexing else deqflow2.deq.DEQSliced
    mod = mod_cls()

    return mod


@pytest.fixture()
@pytest.mark.parametrize("variant", ["tiny", "medium", "large", "huge", "gigantic"])
def mod_deqflow(request, variant, mod_deq) -> deqflow2.DEQFlow:
    variant = deqflow2.Variant(variant)
    mod = deqflow2.DEQFlow(variant=variant, deq=mod_deq)

    return mod

    