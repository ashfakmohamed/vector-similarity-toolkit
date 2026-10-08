from math import sqrt


def _validate(left: list[float], right: list[float]) -> None:
    if not left or len(left) != len(right):
        raise ValueError("vectors must be non-empty and have equal dimensions")


def dot_product(left: list[float], right: list[float]) -> float:
    _validate(left, right)
    return sum(a * b for a, b in zip(left, right, strict=True))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = dot_product(left, right)
    magnitude = sqrt(sum(value * value for value in left)) * sqrt(
        sum(value * value for value in right)
    )
    return dot / magnitude if magnitude else 0.0


def top_k(query: list[float], vectors: dict[str, list[float]], limit: int = 3) -> list[tuple[str, float]]:
    if limit < 1:
        raise ValueError("limit must be positive")
    ranked = ((key, cosine_similarity(query, vector)) for key, vector in vectors.items())
    return sorted(ranked, key=lambda item: (-item[1], item[0]))[:limit]
