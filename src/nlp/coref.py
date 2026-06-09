"""Text-based coreference resolution for EASDRL/CEASDRL-style samples.

Dataset annotations such as ``acts[*].obj_idxs`` are deliberately not used
here. A dataset sample is only converted to text; all antecedent candidates are
then inferred from the text itself.

The implementation is dependency-free in this environment. If spaCy and an
English pipeline are installed later, noun chunks from spaCy are used for
antecedent extraction. Otherwise the module falls back to a conservative
rule-based NLP heuristic.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
import re
from typing import Any, Iterable

from src.utils import load_pkl


DEFAULT_DATA_DIR = os.path.join("data", "easdrl")
DEFAULT_OUTPUT_DIR = os.path.join("data", "coref")
DEFAULT_LLM_OUTPUT_DIR = os.path.join("data", "coref_llm")
DOMAINS = {
    "cooking": "cooking_labeled_text_data.pkl",
    "wikihow": "wikihow_labeled_text_data.pkl",
    "win2k": "win2k_labeled_text_data.pkl",
}
_SPACY_NLP: Any | None = None

_TOKEN_RE = re.compile(r"\w+(?:['’]\w+)?|[^\w\s]")
_PRONOUNS = {
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "this",
    "that",
    "these",
    "those",
}
_SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves"}
_PLURAL_PRONOUNS = {"they", "them", "their", "theirs", "themselves", "these", "those"}
_SINGULAR_PRONOUNS = {"it", "its", "itself", "this", "that"}
_POSSESSIVE_PRONOUNS = {"its", "their", "theirs"}
_DEMONSTRATIVE_PRONOUNS = {"this", "that", "these", "those"}
_GENERIC_MENTIONS = {"one", "ones", "something", "anything", "everything", "nothing", "kind"}
_PRONOUN_FOLLOWERS = {
    "are",
    "be",
    "can",
    "could",
    "did",
    "do",
    "does",
    "has",
    "have",
    "is",
    "may",
    "might",
    "must",
    "should",
    "was",
    "were",
    "will",
    "would",
}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "before",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "have",
    "if",
    "including",
    "in",
    "into",
    "is",
    "may",
    "must",
    "of",
    "off",
    "on",
    "or",
    "some",
    "since",
    "that",
    "the",
    "then",
    "to",
    "was",
    "will",
    "with",
}
_DETERMINERS = {"a", "an", "the", "some", "any", "this", "that", "these", "those", "your"}
_PREPOSITIONS = {
    "at",
    "before",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "off",
    "on",
    "to",
    "with",
}
_ACTION_WORDS = {
    "accept",
    "add",
    "apply",
    "away",
    "be",
    "bring",
    "check",
    "click",
    "coat",
    "combine",
    "combined",
    "contain",
    "contains",
    "come",
    "dispose",
    "donate",
    "drop",
    "keep",
    "leave",
    "make",
    "mix",
    "place",
    "point",
    "pour",
    "power",
    "press",
    "put",
    "recycle",
    "remove",
    "select",
    "serve",
    "set",
    "spray",
    "start",
    "take",
    "throw",
    "turn",
    "use",
    "used",
}
_ADVERBIALS = {"away", "directly", "fully", "gently", "likely", "properly", "right", "well"}
_SENTENCE_END = {".", "!", "?"}


@dataclass(frozen=True)
class Mention:
    """An antecedent candidate in token coordinates."""

    start: int
    end: int
    text: str
    token_indices: tuple[int, ...]
    is_plural: bool = False
    source: str = "text_np"
    sentence: int = 0
    is_subject_like: bool = False


@dataclass(frozen=True)
class Replacement:
    """A single pronoun replacement decision."""

    token_index: int
    pronoun: str
    replacement: str
    antecedent: str
    antecedent_indices: tuple[int, ...]
    source: str


@dataclass(frozen=True)
class CorefResult:
    """Resolved text plus traceable replacement metadata."""

    original_text: str
    resolved_text: str
    replacements: list[Replacement]

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "resolved_text": self.resolved_text,
            "replacements": [asdict(replacement) for replacement in self.replacements],
        }


def load_labeled_dataset(name: str, data_dir: str = DEFAULT_DATA_DIR, limit: int | None = None) -> list[dict[str, Any]]:
    """Load a labeled dataset pkl the same way ``experiment.py`` does."""

    path = os.path.join(data_dir, _normalise_dataset_name(name))
    dataset = load_pkl(path)
    if limit is not None:
        dataset = dataset[:limit]
    return dataset


def resolve_domain(
    domain: str,
    *,
    data_dir: str = DEFAULT_DATA_DIR,
    limit: int | None = None,
    include_second_person: bool = False,
    resolve_singular: bool = False,
) -> list[dict[str, Any]]:
    """Resolve all docs in one dataset domain and keep source provenance."""

    if domain not in DOMAINS:
        raise ValueError(f"unknown domain: {domain}; expected one of {sorted(DOMAINS)}")

    source_file = DOMAINS[domain]
    dataset = load_labeled_dataset(source_file, data_dir=data_dir, limit=limit)
    records = []
    for doc_id, sample in enumerate(dataset):
        result = resolve_sample(
            sample,
            include_second_person=include_second_person,
            resolve_singular=resolve_singular,
        )
        records.append(
            {
                "domain": domain,
                "doc_id": doc_id,
                "source_file": os.path.join(data_dir, source_file),
                "coref": result.to_dict(),
            }
        )
    return records


def resolve_all_domains(
    *,
    data_dir: str = DEFAULT_DATA_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    domains: Iterable[str] = DOMAINS.keys(),
    limit: int | None = None,
    include_second_person: bool = False,
    resolve_singular: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Resolve selected domains and save per-domain plus combined JSON files."""

    os.makedirs(output_dir, exist_ok=True)
    outputs: dict[str, list[dict[str, Any]]] = {}
    all_records = []

    for domain in domains:
        records = resolve_domain(
            domain,
            data_dir=data_dir,
            limit=limit,
            include_second_person=include_second_person,
            resolve_singular=resolve_singular,
        )
        outputs[domain] = records
        all_records.extend(records)
        _write_json(os.path.join(output_dir, f"{domain}_coref.json"), records)

    _write_json(os.path.join(output_dir, "all_domains_coref.json"), all_records)
    return outputs


def resolve_text_with_llm(text: str, model_name: str, *, temperature: float = 0) -> dict[str, Any]:
    """Resolve coreferences by asking an LLM to return only rewritten text."""

    from src.llm import generate_prompt, generate_responses

    prompt = generate_prompt("llm_coref_resolution", {"nl": text})
    response = generate_responses(model_name, prompt, temperature=temperature, log=True)
    resolved_text = _clean_llm_text_response(response["content"])
    return {
        "original_text": text,
        "resolved_text": resolved_text,
        "model": response.get("model", model_name),
        "response_time": response.get("response_time"),
        "usage": response.get("usage"),
    }


def resolve_sample_with_llm(sample: dict[str, Any], model_name: str, *, temperature: float = 0) -> dict[str, Any]:
    """Resolve one dataset sample with the LLM prompt."""

    return resolve_text_with_llm(sample_to_text(sample), model_name, temperature=temperature)


def resolve_domain_with_llm(
    domain: str,
    model_name: str,
    *,
    data_dir: str = DEFAULT_DATA_DIR,
    output_dir: str = DEFAULT_LLM_OUTPUT_DIR,
    limit: int | None = None,
    temperature: float = 0,
) -> list[dict[str, Any]]:
    """Resolve all docs in one domain with an LLM and save a JSONL file."""

    if domain not in DOMAINS:
        raise ValueError(f"unknown domain: {domain}; expected one of {sorted(DOMAINS)}")

    os.makedirs(output_dir, exist_ok=True)
    source_file = DOMAINS[domain]
    dataset = load_labeled_dataset(source_file, data_dir=data_dir, limit=limit)
    records = []
    output_path = os.path.join(output_dir, f"{domain}_llm_coref.jsonl")

    with open(output_path, "w", encoding="utf-8") as file:
        for doc_id, sample in enumerate(dataset):
            coref = resolve_sample_with_llm(sample, model_name, temperature=temperature)
            record = {
                "domain": domain,
                "doc_id": doc_id,
                "source_file": os.path.join(data_dir, source_file),
                "model": model_name,
                **coref,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            records.append(record)
    return records


def resolve_all_domains_with_llm(
    model_name: str,
    *,
    data_dir: str = DEFAULT_DATA_DIR,
    output_dir: str = DEFAULT_LLM_OUTPUT_DIR,
    domains: Iterable[str] = DOMAINS.keys(),
    limit: int | None = None,
    temperature: float = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Resolve selected domains with an LLM and save per-domain plus combined JSONL files."""

    os.makedirs(output_dir, exist_ok=True)
    outputs: dict[str, list[dict[str, Any]]] = {}
    all_path = os.path.join(output_dir, "all_domains_llm_coref.jsonl")

    with open(all_path, "w", encoding="utf-8") as all_file:
        for domain in domains:
            records = resolve_domain_with_llm(
                domain,
                model_name,
                data_dir=data_dir,
                output_dir=output_dir,
                limit=limit,
                temperature=temperature,
            )
            outputs[domain] = records
            for record in records:
                all_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            all_file.flush()
    return outputs


def sample_to_text(sample: dict[str, Any]) -> str:
    """Convert an EASDRL sample to paragraph text like ``experiment.py``."""

    if "sents" in sample:
        return ". ".join(" ".join(str(token) for token in sent) for sent in sample["sents"]) + "."
    if "words" in sample:
        return _detokenize([str(token) for token in sample["words"]])
    raise ValueError("sample must contain either 'sents' or 'words'")


def resolve_coreferences(
    text_or_sample: str | dict[str, Any],
    *,
    include_second_person: bool = False,
    resolve_singular: bool = True,
) -> CorefResult:
    """Resolve pronouns in raw text or in one dataset sample.

    No dataset annotation fields are consulted. For samples, only ``sents`` or
    ``words`` are used to build the input text.
    """

    if isinstance(text_or_sample, dict):
        return resolve_sample(
            text_or_sample,
            include_second_person=include_second_person,
            resolve_singular=resolve_singular,
        )
    return resolve_text(
        text_or_sample,
        include_second_person=include_second_person,
        resolve_singular=resolve_singular,
    )


def resolve_sample(
    sample: dict[str, Any],
    *,
    include_second_person: bool = False,
    resolve_singular: bool = True,
) -> CorefResult:
    """Resolve coreferences in one EASDRL/CEASDRL sample dict using only text."""

    return resolve_text(
        sample_to_text(sample),
        include_second_person=include_second_person,
        resolve_singular=resolve_singular,
    )


def resolve_text(
    text: str,
    *,
    include_second_person: bool = False,
    resolve_singular: bool = True,
) -> CorefResult:
    """Resolve coreferences in raw text using text-derived antecedents only."""

    spacy_result = _try_resolve_with_spacy(
        text,
        include_second_person=include_second_person,
        resolve_singular=resolve_singular,
    )
    if spacy_result is not None:
        return spacy_result

    tokens = _TOKEN_RE.findall(text)
    mentions = _extract_rule_based_mentions(tokens)
    return _resolve_tokens(
        tokens,
        mentions,
        include_second_person=include_second_person,
        resolve_singular=resolve_singular,
    )


def resolve_dataset(
    dataset: Iterable[dict[str, Any]],
    *,
    include_second_person: bool = False,
    resolve_singular: bool = True,
    in_place: bool = False,
) -> list[dict[str, Any]]:
    """Resolve every sample in a dataset and attach a ``coref`` result dict."""

    resolved = []
    for sample in dataset:
        target = sample if in_place else dict(sample)
        target["coref"] = resolve_sample(
            target,
            include_second_person=include_second_person,
            resolve_singular=resolve_singular,
        ).to_dict()
        resolved.append(target)
    return resolved


def extract_entity_mentions(text_or_sample: str | dict[str, Any]) -> list[Mention]:
    """Extract text-derived antecedent candidates.

    This function is kept as a public helper, but unlike the previous version it
    does not read ``acts`` or any other annotation field.
    """

    text = sample_to_text(text_or_sample) if isinstance(text_or_sample, dict) else text_or_sample
    tokens = _TOKEN_RE.findall(text)
    return _extract_rule_based_mentions(tokens)


def _try_resolve_with_spacy(
    text: str,
    *,
    include_second_person: bool,
    resolve_singular: bool,
) -> CorefResult | None:
    try:
        import spacy  # type: ignore
    except Exception:
        return None

    nlp = _load_spacy_pipeline(spacy)
    if nlp is None:
        return None

    doc = nlp(text)
    tokens = [token.text for token in doc]
    sent_ids = {sent.start: sent_id for sent_id, sent in enumerate(doc.sents)}
    mentions = [
        Mention(
            start=chunk.start,
            end=chunk.end,
            text=chunk.text,
            token_indices=tuple(range(chunk.start, chunk.end)),
            is_plural=_spacy_chunk_is_plural(chunk),
            source="spacy_example_noun_chunk" if _is_example_context(tokens, chunk.start) else "spacy_noun_chunk",
            sentence=sent_ids.get(chunk.sent.start, 0),
            is_subject_like=chunk.root.dep_ in {"nsubj", "nsubjpass"},
        )
        for chunk in doc.noun_chunks
        if _is_usable_spacy_chunk(chunk)
    ]
    if not mentions:
        return None
    return _resolve_tokens(
        tokens,
        mentions,
        include_second_person=include_second_person,
        resolve_singular=resolve_singular,
    )


def _load_spacy_pipeline(spacy: Any) -> Any | None:
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP

    for name in ("en_core_web_trf", "en_core_web_lg", "en_core_web_md", "en_core_web_sm"):
        try:
            _SPACY_NLP = spacy.load(name)
            return _SPACY_NLP
        except Exception:
            continue
    return None


def _resolve_tokens(
    tokens: list[str],
    mentions: list[Mention],
    *,
    include_second_person: bool,
    resolve_singular: bool,
) -> CorefResult:
    original_text = _detokenize(tokens)
    pronouns = set(_PRONOUNS)
    if not resolve_singular:
        pronouns.difference_update(_SINGULAR_PRONOUNS)
        pronouns.difference_update({"its", "itself"})
    if include_second_person:
        pronouns.update(_SECOND_PERSON)

    replacements: list[Replacement] = []
    resolved_tokens = list(tokens)
    active_mentions = sorted(mentions, key=lambda mention: (mention.end, mention.start))

    for index, token in enumerate(tokens):
        lower = token.lower()
        if lower not in pronouns or not _is_replaceable_pronoun(tokens, index):
            continue

        antecedent = _choose_antecedent(lower, index, active_mentions, tokens)
        if antecedent is None:
            continue

        replacement = _replacement_text(token, antecedent.text, lower in _POSSESSIVE_PRONOUNS)
        resolved_tokens[index] = replacement
        replacements.append(
            Replacement(
                token_index=index,
                pronoun=token,
                replacement=replacement,
                antecedent=antecedent.text,
                antecedent_indices=antecedent.token_indices,
                source=antecedent.source,
            )
        )
        if antecedent.is_plural:
            active_mentions.append(
                Mention(
                    start=index,
                    end=index + 1,
                    text=antecedent.text,
                    token_indices=antecedent.token_indices,
                    is_plural=antecedent.is_plural,
                    source="resolved_pronoun",
                    sentence=_sentence_for_start(index, _sentence_starts(tokens)),
                )
            )
            active_mentions.sort(key=lambda mention: (mention.end, mention.start))

    return CorefResult(
        original_text=original_text,
        resolved_text=_detokenize(resolved_tokens),
        replacements=replacements,
    )


def _choose_antecedent(pronoun: str, token_index: int, mentions: list[Mention], tokens: list[str]) -> Mention | None:
    if pronoun in {"this", "that"}:
        return None
    if pronoun == "it" and _is_pleonastic_or_abstract_it(tokens, token_index):
        return None

    previous = [
        mention
        for mention in mentions
        if mention.end <= token_index and _is_usable_mention(mention)
    ]
    if not previous:
        return None

    wants_plural = pronoun in _PLURAL_PRONOUNS
    wants_singular = pronoun in _SINGULAR_PRONOUNS
    compatible = [
        mention
        for mention in previous
        if (wants_plural and mention.is_plural) or (wants_singular and not mention.is_plural)
    ]
    candidates = compatible
    if not candidates:
        return None

    pronoun_sentence = _sentence_for_start(token_index, _sentence_starts(tokens))
    pronoun_at_sentence_start = _is_at_sentence_start(tokens, token_index)

    if wants_plural:
        candidates = _plural_candidates(
            candidates,
            pronoun_sentence,
            pronoun_at_sentence_start,
            token_index,
            tokens,
        )
    elif wants_singular:
        candidates = _singular_candidates(candidates, pronoun_sentence, token_index)
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda mention: _mention_score(mention, token_index, pronoun_sentence, pronoun_at_sentence_start),
    )


def _plural_candidates(
    candidates: list[Mention],
    pronoun_sentence: int,
    pronoun_at_sentence_start: bool,
    token_index: int,
    tokens: list[str],
) -> list[Mention]:
    non_example = [mention for mention in candidates if "example" not in mention.source]
    candidates = non_example or candidates

    if pronoun_at_sentence_start:
        if _previous_sentence_is_demonstrative_generic(tokens, pronoun_sentence):
            shifted = [
                mention
                for mention in candidates
                if mention.sentence == pronoun_sentence - 2
                and mention.source != "resolved_pronoun"
                and "example" not in mention.source
            ]
            if shifted:
                return shifted

        recent = [mention for mention in candidates if 0 <= pronoun_sentence - mention.sentence <= 2]
        previous_subjects = [
            mention for mention in recent if mention.sentence == pronoun_sentence - 1 and mention.is_subject_like
        ]
        if previous_subjects:
            return previous_subjects
        if recent:
            return recent

    return [
        mention
        for mention in candidates
        if token_index - mention.end <= 80
        and (mention.source != "resolved_pronoun" or token_index - mention.end <= 40)
    ]


def _singular_candidates(candidates: list[Mention], pronoun_sentence: int, token_index: int) -> list[Mention]:
    candidates = [
        mention
        for mention in candidates
        if "example" not in mention.source
        and mention.source != "resolved_pronoun"
        and _normalised_mention_text(mention.text) not in _GENERIC_MENTIONS
    ]
    return [
        mention
        for mention in candidates
        if (
            mention.sentence == pronoun_sentence and token_index - mention.end <= 12
        )
        or (
            mention.sentence == pronoun_sentence - 1 and token_index - mention.end <= 30
        )
    ]


def _mention_score(
    mention: Mention,
    token_index: int,
    pronoun_sentence: int,
    pronoun_at_sentence_start: bool,
) -> tuple[int, int, int]:
    distance = token_index - mention.end
    subject_bonus = 30 if (
        pronoun_at_sentence_start and mention.is_subject_like and mention.sentence == pronoun_sentence - 1
    ) else 0
    source_bonus = 40 if mention.source == "resolved_pronoun" else 0
    source_penalty = 20 if "example" in mention.source else 0
    length_bonus = min(mention.end - mention.start, 4)
    return (-distance + subject_bonus + source_bonus - source_penalty, length_bonus, mention.start)


def _extract_rule_based_mentions(tokens: list[str]) -> list[Mention]:
    mentions: list[Mention] = []
    current_start: int | None = None
    sentence_starts = _sentence_starts(tokens)

    for index, token in enumerate(tokens + ["."]):
        if _is_np_token(token):
            if current_start is None:
                current_start = index
            continue

        if current_start is not None:
            _append_mention(mentions, tokens, current_start, index, "rule_np", sentence_starts)
            current_start = None

    _append_coordinate_mentions(mentions, tokens, sentence_starts)
    return _dedupe_mentions(mentions)


def _append_coordinate_mentions(mentions: list[Mention], tokens: list[str], sentence_starts: list[int]) -> None:
    by_end = {mention.end: mention for mention in mentions}
    by_start = {mention.start: mention for mention in mentions}
    for index, token in enumerate(tokens):
        if token.lower() not in {"and", "or"}:
            continue
        left = by_end.get(index)
        right = by_start.get(index + 1)
        if left is None or right is None:
            continue
        text = _detokenize(tokens[left.start : right.end])
        mentions.append(
            Mention(
                start=left.start,
                end=right.end,
                text=text,
                token_indices=tuple(range(left.start, right.end)),
                is_plural=True,
                source="rule_example_np" if _is_example_context(tokens, left.start) else "rule_coord_np",
                sentence=_sentence_for_start(left.start, sentence_starts),
                is_subject_like=_is_subject_like(tokens, left.start, sentence_starts),
            )
        )


def _append_mention(
    mentions: list[Mention],
    tokens: list[str],
    start: int,
    end: int,
    source: str,
    sentence_starts: list[int],
) -> None:
    while start < end and tokens[start].lower() in _DETERMINERS:
        start += 1
    while end > start and tokens[end - 1].lower() in _PREPOSITIONS:
        end -= 1
    if end <= start:
        return

    text = _detokenize(tokens[start:end])
    if not text or _is_pronoun_text(text):
        return
    mentions.append(
        Mention(
            start=start,
            end=end,
            text=text,
            token_indices=tuple(range(start, end)),
            is_plural=_looks_plural(text, end - start > 1),
            source="rule_example_np" if _is_example_context(tokens, start) else source,
            sentence=_sentence_for_start(start, sentence_starts),
            is_subject_like=_is_subject_like(tokens, start, sentence_starts),
        )
    )


def _dedupe_mentions(mentions: list[Mention]) -> list[Mention]:
    seen: set[tuple[int, int, str]] = set()
    result = []
    for mention in sorted(mentions, key=lambda item: (item.start, item.end)):
        key = (mention.start, mention.end, mention.text.lower())
        if key in seen:
            continue
        seen.add(key)
        result.append(mention)
    return result


def _is_np_token(token: str) -> bool:
    lower = token.lower()
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9'-]*$", token):
        return False
    if lower in _PRONOUNS or lower in _SECOND_PERSON:
        return False
    if lower in _STOPWORDS or lower in _ACTION_WORDS or lower in _ADVERBIALS:
        return False
    if lower.endswith("ly"):
        return False
    return True


def _is_usable_spacy_chunk(chunk: Any) -> bool:
    text = chunk.text.strip()
    normalised = _normalised_mention_text(text)
    if not text or _is_pronoun_text(text) or normalised in _GENERIC_MENTIONS:
        return False
    if not re.search(r"[A-Za-z0-9]", text):
        return False
    if chunk[0].lower_ in _DEMONSTRATIVE_PRONOUNS:
        return False
    if chunk.root.pos_ not in {"NOUN", "PROPN", "NUM"}:
        return False
    return True


def _spacy_chunk_is_plural(chunk: Any) -> bool:
    if any(token.lower_ in {"and", "or"} for token in chunk):
        return True
    number = chunk.root.morph.get("Number")
    if "Plur" in number:
        return True
    if chunk.root.tag_ in {"NNS", "NNPS"}:
        return True
    return False


def _is_usable_mention(mention: Mention) -> bool:
    text = mention.text.strip()
    normalised = _normalised_mention_text(text)
    if not text or normalised in _GENERIC_MENTIONS:
        return False
    if not re.search(r"[A-Za-z0-9]", text):
        return False
    if text.split()[0].lower() in _DEMONSTRATIVE_PRONOUNS:
        return False
    return True


def _is_pleonastic_or_abstract_it(tokens: list[str], index: int) -> bool:
    next_token = tokens[index + 1].lower() if index + 1 < len(tokens) else ""
    next_next = tokens[index + 2].lower() if index + 2 < len(tokens) else ""
    if next_token in {"'s", "is", "was"} and next_next in {
        "best",
        "better",
        "important",
        "possible",
        "necessary",
        "easy",
        "hard",
        "generally",
        "likely",
    }:
        return True
    return False


def _normalised_mention_text(text: str) -> str:
    return re.sub(r"^\W+|\W+$", "", text.strip().lower())


def _sentence_starts(tokens: list[str]) -> list[int]:
    starts = [0]
    for index, token in enumerate(tokens[:-1]):
        if token in _SENTENCE_END:
            starts.append(index + 1)
    return starts


def _sentence_for_start(start: int, sentence_starts: list[int]) -> int:
    sentence = 0
    for index, sentence_start in enumerate(sentence_starts):
        if sentence_start <= start:
            sentence = index
        else:
            break
    return sentence


def _is_subject_like(tokens: list[str], start: int, sentence_starts: list[int]) -> bool:
    sentence_start = sentence_starts[_sentence_for_start(start, sentence_starts)]
    before = [token.lower() for token in tokens[sentence_start:start]]
    return not any(token in _ACTION_WORDS or token in _PREPOSITIONS for token in before)


def _is_at_sentence_start(tokens: list[str], index: int) -> bool:
    return index == 0 or tokens[index - 1] in _SENTENCE_END


def _previous_sentence_is_demonstrative_generic(tokens: list[str], pronoun_sentence: int) -> bool:
    starts = _sentence_starts(tokens)
    if pronoun_sentence <= 0 or pronoun_sentence - 1 >= len(starts):
        return False
    start = starts[pronoun_sentence - 1]
    while start < len(tokens) and tokens[start] in {".", "!", "?"}:
        start += 1
    first = tokens[start].lower() if start < len(tokens) else ""
    second = tokens[start + 1].lower() if start + 1 < len(tokens) else ""
    return first in {"this", "that"} and second in {"kind", "type", "sort"}


def _is_example_context(tokens: list[str], start: int) -> bool:
    context = [token.lower() for token in tokens[max(0, start - 3) : start]]
    return "such" in context or "like" in context or "including" in context


def _is_replaceable_pronoun(tokens: list[str], index: int) -> bool:
    pronoun = tokens[index].lower()
    if pronoun not in _DEMONSTRATIVE_PRONOUNS:
        return True
    next_token = tokens[index + 1].lower() if index + 1 < len(tokens) else ""
    prev_token = tokens[index - 1] if index > 0 else "."
    at_clause_start = index == 0 or prev_token in _SENTENCE_END or prev_token in {";", ":"}
    return at_clause_start and (not next_token or next_token in _PRONOUN_FOLLOWERS)


def _replacement_text(pronoun: str, antecedent: str, possessive: bool) -> str:
    replacement = antecedent
    if possessive:
        replacement = antecedent + ("'" if antecedent.endswith("s") else "'s")
    if pronoun[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _looks_plural(text: str, multiple_tokens: bool) -> bool:
    lower = text.lower()
    if " and " in lower or "," in lower:
        return True
    last = re.sub(r"^\W+|\W+$", "", lower.split()[-1]) if lower.split() else lower
    if last.endswith(("ies", "ses", "xes", "zes", "ches", "shes")):
        return True
    if last.endswith("s") and not last.endswith(("ss", "us")):
        return True
    return False


def _is_pronoun_text(text: str) -> bool:
    return text.strip().lower() in _PRONOUNS or text.strip().lower() in _SECOND_PERSON


def _detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,;:!?%])", r"\1", text)
    text = re.sub(r"([({\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)}\]])", r"\1", text)
    text = text.replace(" n't", "n't")
    text = text.replace(" 's", "'s")
    text = text.replace(" ’s", "'s")
    return text


def _normalise_dataset_name(name: str) -> str:
    if name in DOMAINS:
        return DOMAINS[name]
    if name.endswith(".pkl"):
        return name
    return f"{name}.pkl"


def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _clean_llm_text_response(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:text)?\s*([\s\S]*?)\s*```", text, re.I)
    if fence:
        text = fence.group(1).strip()
    for prefix in ("Rewritten paragraph:", "Rewritten text:", "Output:", "TEXT:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    return text.strip().strip('"')


def main() -> None:
    parser = argparse.ArgumentParser(description="Run text-based coreference resolution on EASDRL domains.")
    parser.add_argument(
        "--method",
        choices=["nlp", "llm"],
        default="nlp",
        help="Use local spaCy/rule-based coref or LLM rewriting.",
    )
    parser.add_argument(
        "-d",
        "--domains",
        nargs="+",
        default=list(DOMAINS),
        choices=list(DOMAINS),
        help="Domains to process.",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory containing *_labeled_text_data.pkl files.")
    parser.add_argument("--output-dir", help="Directory for coref outputs.")
    parser.add_argument("-l", "--limit", type=int, help="Limit docs per domain for debugging.")
    parser.add_argument("-m", "--model", default="gpt-4.1-mini", help="LLM model for --method llm.")
    parser.add_argument("-t", "--temperature", type=float, default=0, help="LLM temperature for --method llm.")
    parser.add_argument("--include-second-person", action="store_true", help="Also resolve you/your pronouns.")
    parser.add_argument(
        "--resolve-singular",
        action="store_true",
        help="Also resolve singular it/its/itself. Off by default for higher precision on datasets.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (DEFAULT_LLM_OUTPUT_DIR if args.method == "llm" else DEFAULT_OUTPUT_DIR)
    if args.method == "llm":
        outputs = resolve_all_domains_with_llm(
            args.model,
            data_dir=args.data_dir,
            output_dir=output_dir,
            domains=args.domains,
            limit=args.limit,
            temperature=args.temperature,
        )
    else:
        outputs = resolve_all_domains(
            data_dir=args.data_dir,
            output_dir=output_dir,
            domains=args.domains,
            limit=args.limit,
            include_second_person=args.include_second_person,
            resolve_singular=args.resolve_singular,
        )
    for domain, records in outputs.items():
        if args.method == "llm":
            print(f"{domain}: {len(records)} docs rewritten by {args.model}")
        else:
            replacements = sum(len(record["coref"]["replacements"]) for record in records)
            print(f"{domain}: {len(records)} docs, {replacements} replacements")
    print(f"Saved coref outputs to {output_dir}")


if __name__ == "__main__":
    main()
