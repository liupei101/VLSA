from .plain_prompt_learner import PlainPromptLearner
from .rank_prompt_learner import RankPromptLearner
from .prompt_adapter import PromptAdapter


def load_prompt_learner(learner_name: str, cfg):
    if learner_name == 'plain':
        prompt_learner = PlainPromptLearner(**cfg)

    elif learner_name == 'rank':
        prompt_learner = RankPromptLearner(**cfg)

    else:
        prompt_learner = None

    print(f"A prompt learner of {learner_name} is loaded.")
    return prompt_learner

def load_prompt_adapter(prompt_encoder, cfg):
    adapter = PromptAdapter(prompt_encoder, **cfg)
    print(f"A prompt adapter is loaded.")
    return adapter
