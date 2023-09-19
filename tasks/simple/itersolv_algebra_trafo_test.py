from .simple_task import SimpleTask
from .transformer_classifier_mixin import TransformerClassifierMixin
from .itersolv_algebra_test_mixin import IterSolvAlgebraTestMixin
from .. import task, args
import framework


@task()
class ItersolvAlgebraTrafoTest(IterSolvAlgebraTestMixin, TransformerClassifierMixin, SimpleTask):
    pass