import re
from vocab import load_vocab, vocab_fraction

VOCAB = load_vocab(n=130) - set(
    [
        "that",
        "the",
        "would",
        "be",
        "are",
        "is",
        "a",
        "an",
        "and",
        "of",
        "to",
        "in",
        "for",
        "on",
        "with",
        "as",
        "by",
        "at",
        "from",
        "i",
        "1",
    ]
)

print(VOCAB)


def reward_vocab(
    completions: list[str],
    **kwargs,
) -> list[float]:
    scores = []

    for completion in completions:
        tokens = re.findall(r"[a-z]+", completion.lower())
        length = len(tokens)

        vocab_score = vocab_fraction(completion, VOCAB) * 0.3

        if length == 0:
            brevity_score = 0.0
        else:
            brevity_score = min(1.0, 10 / length) * 0.2

        score = vocab_score + brevity_score
        scores.append(score)

    return scores


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def reward_correctness(
    completions: list[str],
    answer_aliases: list[list[str]],
    **kwargs,
) -> list[float]:
    scores = []
    for completion, aliases in zip(completions, answer_aliases):
        norm_c = _normalise(completion)
        score = 0.0
        for alias in aliases:
            norm_a = _normalise(alias)
            if not norm_a:
                continue
            if norm_a == norm_c:
                score = 1.0
                break
            if norm_a in norm_c:
                score = max(score, 0.5)
        scores.append(score * 1.6)
    return scores
