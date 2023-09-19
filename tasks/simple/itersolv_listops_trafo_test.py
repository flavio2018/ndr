from .simple_task import SimpleTask
from .transformer_classifier_mixin import TransformerClassifierMixin
from .itersolv_listops_test_mixin import IterSolvListopsTestMixin
from .. import task, args
import framework


@task()
class ItersolvListopsTrafoTest(IterSolvListopsTestMixin, TransformerClassifierMixin, SimpleTask):
    pass