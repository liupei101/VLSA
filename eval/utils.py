from .evaluator_clf import BinClf_Evaluator
from .evaluator_clf import MultiClf_Evaluator
from .evaluator_surv import RegSurv_Evaluator
from .evaluator_surv import NLLSurv_Evaluator
from .evaluator_surv import CoxSurv_Evaluator

def load_evaluator(task, *args, **kws):
    if task == 'clf':
        if args[0] == 'Binary':
            evaluator = BinClf_Evaluator(**kws)
        elif args[0] == 'Multi-class':
            evaluator = MultiClf_Evaluator(**kws)
        else:
            evaluator = None
    elif task == 'sa':
        if args[0] == 'Reg':
            evaluator = RegSurv_Evaluator(**kws)
        elif args[0] == 'NLL':
            evaluator = NLLSurv_Evaluator(prediction_type='hazard', **kws)
        elif args[0] == 'NLL-IF':
            evaluator = NLLSurv_Evaluator(prediction_type='incidence', **kws)
        elif args[0] == 'Cox':
            evaluator = CoxSurv_Evaluator(**kws)
        else:
            evaluator = None
    elif task == 'vlsa':
        if args[0] == 'VL':
            evaluator = NLLSurv_Evaluator(prediction_type='hazard', **kws)
        elif args[0] == 'VL-IF':
            evaluator = NLLSurv_Evaluator(prediction_type='incidence', **kws)
        else:
            evaluator = None
    else:
        pass
    
    return evaluator
