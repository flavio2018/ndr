from .simple_task import SimpleTask
from .transformer_classifier_mixin import TransformerClassifierMixin
from .itersolv_arithmetic_test_mixin import IterSolvArithmeticTestMixin
from .. import task


@task
class ItersolvArithmeticTrafoTest(IterSolvArithmeticTestMixin, TransformerClassifierMixin, SimpleTask):
    pass