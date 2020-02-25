import numpy as np

from recogym import build_agent_init
from recogym.agents import LogregPolyAgent, logreg_poly_args


agent = build_agent_init(
    'LikelihoodWithTime',
    LogregPolyAgent,
    {
        **logreg_poly_args,
        'weight_history_function': lambda t: np.exp(-t)
    }
)
