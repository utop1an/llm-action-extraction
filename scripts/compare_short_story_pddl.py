"""Compare short-story NL2P event mentions with hand-authored PDDL actions.

The comparison is deliberately descriptive: NL2P produces ordered event
mentions, while a domain PDDL defines reusable action schemas.  The script
therefore reports strict name overlap, lightweight lexical correspondences,
and coarse parameter/object-type distributions without treating the PDDL as
an event-by-event gold annotation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NL2P_DIR = ROOT / "results/short_stories/nl2p_1/gpt-5"
DEFAULT_PDDL_DIR = ROOT / "data/short_stories/mulab_short_stories/pddls"
DEFAULT_OUTPUT_DIR = ROOT / "results/short_stories/domain_comparison"

STORIES = {
    "CHIC": ("Chicken-Licken", "chicken-licken.json"),
    "HANS": ("Hansel and Gretel", "hansel-and-gretel.json"),
    "JACK": ("Jack and the Beanstalk", "jack-and-the-beanstalk.json"),
    "REMA": ("The Three Remarks", "the-three-remarks.json"),
    "PIGS": ("The Three Little Pigs", "the-three-little-pigs.json"),
    "FOUR": ("The Four Skillful Brothers", "the-four-skillful-brothers.json"),
}

STOP_WORDS = {
    "a", "an", "and", "at", "be", "by", "for", "from", "in", "into",
    "of", "on", "the", "to", "with", "again", "character", "object",
}
ROLE_WORDS = {
    "boy", "brother", "brothers", "child", "children", "dragon", "duck",
    "father", "fox", "giant", "hen", "impostor", "king", "mother", "pig",
    "pigs", "princess", "robber", "robbers", "sons", "suitor", "suitors",
    "wife", "witch", "wolf",
}
IRREGULAR = {
    "ate": "eat", "eaten": "eat", "bought": "buy", "brought": "bring",
    "built": "build",
    "came": "come", "fell": "fall", "fled": "flee", "gave": "give",
    "got": "get", "grew": "grow", "heard": "hear", "hid": "hide",
    "left": "leave", "met": "meet", "ran": "run", "rode": "ride",
    "shot": "shoot", "slept": "sleep", "sold": "sell", "spoke": "speak",
    "stole": "steal", "swam": "swim", "took": "take", "told": "tell",
    "went": "go", "woke": "wake",
}


@dataclass(frozen=True)
class PDDLAction:
    name: str
    parameter_types: tuple[str, ...]


@dataclass(frozen=True)
class PDDLDomain:
    name: str
    types: tuple[str, ...]
    constants: dict[str, str]
    actions: tuple[PDDLAction, ...]


def _tokenize_pddl(text: str) -> list[str]:
    text = re.sub(r";[^\n]*", "", text)
    return re.findall(r"\(|\)|[^\s()]+", text.lower())


def _parse_sexpr(tokens: list[str]) -> list[Any]:
    stack: list[list[Any]] = []
    roots: list[Any] = []
    for token in tokens:
        if token == "(":
            node: list[Any] = []
            if stack:
                stack[-1].append(node)
            else:
                roots.append(node)
            stack.append(node)
        elif token == ")":
            if not stack:
                raise ValueError("Unexpected ')' in PDDL")
            stack.pop()
        else:
            if not stack:
                raise ValueError(f"Token outside a list: {token}")
            stack[-1].append(token)
    if stack:
        raise ValueError("Unclosed '(' in PDDL")
    return roots


def _typed_symbols(items: Iterable[Any], variable_only: bool = False) -> dict[str, str]:
    flat = [item for item in items if isinstance(item, str)]
    result: dict[str, str] = {}
    pending: list[str] = []
    index = 0
    while index < len(flat):
        token = flat[index]
        if token == "-" and index + 1 < len(flat):
            symbol_type = flat[index + 1]
            for symbol in pending:
                if not variable_only or symbol.startswith("?"):
                    result[symbol] = symbol_type
            pending = []
            index += 2
        else:
            pending.append(token)
            index += 1
    for symbol in pending:
        if not variable_only or symbol.startswith("?"):
            result[symbol] = "object"
    return result


def parse_domain(path: Path) -> PDDLDomain:
    roots = _parse_sexpr(_tokenize_pddl(path.read_text(encoding="utf-8")))
    if len(roots) != 1 or not roots[0] or roots[0][0] != "define":
        raise ValueError(f"Expected one PDDL define form in {path}")
    forms = roots[0][1:]
    domain_name = path.stem
    types: list[str] = []
    constants: dict[str, str] = {}
    actions: list[PDDLAction] = []
    for form in forms:
        if not isinstance(form, list) or not form:
            continue
        if form[0] == "domain" and len(form) > 1:
            domain_name = form[1]
        elif form[0] == ":types":
            types.extend(_typed_symbols(form[1:]).keys())
        elif form[0] == ":constants":
            constants.update(_typed_symbols(form[1:]))
        elif form[0] == ":action" and len(form) > 1:
            parameters: tuple[str, ...] = ()
            if ":parameters" in form:
                parameter_index = form.index(":parameters") + 1
                parameter_form = form[parameter_index]
                if isinstance(parameter_form, list):
                    typed = _typed_symbols(parameter_form, variable_only=True)
                    parameters = tuple(typed.values())
            actions.append(PDDLAction(str(form[1]), parameters))
    return PDDLDomain(
        domain_name,
        tuple(dict.fromkeys(types)),
        constants,
        tuple(actions),
    )


def word_tokens(text: str, *, remove_roles: bool = False) -> tuple[str, ...]:
    words = re.findall(r"[a-z0-9]+", text.casefold().replace("_", " "))
    normalized: list[str] = []
    for word in words:
        word = IRREGULAR.get(word, word)
        if len(word) > 4 and word.endswith("ing"):
            word = word[:-3]
        elif len(word) > 3 and word.endswith("ed"):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        word = IRREGULAR.get(word, word)
        if word in STOP_WORDS or (remove_roles and word in ROLE_WORDS):
            continue
        normalized.append(word)
    return tuple(normalized)


def normalized_name(text: str) -> str:
    return "_".join(word_tokens(text))


def lexical_similarity(left: str, right: str) -> float:
    left_tokens = set(word_tokens(left))
    right_tokens = set(word_tokens(right, remove_roles=True))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return 2 * overlap / (len(left_tokens) + len(right_tokens))


def _constant_score(argument: str, constant: str) -> float:
    argument_tokens = set(word_tokens(argument))
    constant_tokens = set(word_tokens(constant))
    if not argument_tokens or not constant_tokens:
        return 0.0
    if constant_tokens <= argument_tokens or argument_tokens <= constant_tokens:
        return 1.0
    overlap = len(argument_tokens & constant_tokens)
    return 2 * overlap / (len(argument_tokens) + len(constant_tokens))


def infer_argument_type(argument: str, domain: PDDLDomain) -> tuple[str, str, float]:
    scored = [
        (_constant_score(argument, constant), constant, object_type)
        for constant, object_type in domain.constants.items()
    ]
    scored.sort(reverse=True)
    if scored and scored[0][0] >= 0.5:
        score, constant, object_type = scored[0]
        tied_types = {item[2] for item in scored if item[0] == score}
        if len(tied_types) == 1:
            return object_type, constant, score

    # A repeated noun such as "house" may match several constants but still
    # imply one unambiguous PDDL type (all house constants are locations).
    argument_tokens = set(word_tokens(argument))
    token_types: dict[str, set[str]] = defaultdict(set)
    for constant, object_type in domain.constants.items():
        for token in word_tokens(constant):
            token_types[token].add(object_type)
    candidate_types = {
        next(iter(token_types[token]))
        for token in argument_tokens
        if token in token_types and len(token_types[token]) == 1
    }
    if len(candidate_types) == 1:
        return next(iter(candidate_types)), "", 0.4
    return "unknown", "", 0.0


def type_overlap(inferred: Iterable[str], expected: Iterable[str]) -> float | None:
    inferred_counter = Counter(item for item in inferred if item != "unknown")
    expected_counter = Counter(expected)
    if not inferred_counter or not expected_counter:
        return None
    overlap = sum((inferred_counter & expected_counter).values())
    return 2 * overlap / (sum(inferred_counter.values()) + sum(expected_counter.values()))


def _best_pddl_action(verb: str, actions: Iterable[PDDLAction]) -> tuple[PDDLAction, float]:
    ranked = sorted(
        ((lexical_similarity(verb, action.name), action) for action in actions),
        key=lambda item: (item[0], item[1].name),
        reverse=True,
    )
    if not ranked:
        raise ValueError("PDDL domain contains no actions")
    score, action = ranked[0]
    return action, score


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_story(
    story: str, payload: dict[str, Any], domain: PDDLDomain
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Counter[str], Counter[str]]:
    occurrences: list[dict[str, Any]] = []
    verb_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    nl2p_type_counts: Counter[str] = Counter()
    for index, record in enumerate(payload["actions"], start=1):
        verb = record["verb"]
        inferred = [infer_argument_type(argument, domain) for argument in record["arguments"]]
        inferred_types = [item[0] for item in inferred]
        nl2p_type_counts.update(inferred_types)
        best_action, similarity = _best_pddl_action(verb, domain.actions)
        occurrence = {
            "story": story,
            "event_index": index,
            "nl2p_verb": verb,
            "nl2p_arguments": " | ".join(record["arguments"]),
            "inferred_argument_types": " | ".join(inferred_types),
            "matched_constants": " | ".join(item[1] for item in inferred),
            "best_pddl_action": best_action.name,
            "name_similarity": round(similarity, 3),
            "pddl_parameter_types": " | ".join(best_action.parameter_types),
            "type_overlap": (
                "" if type_overlap(inferred_types, best_action.parameter_types) is None
                else round(type_overlap(inferred_types, best_action.parameter_types) or 0.0, 3)
            ),
        }
        occurrences.append(occurrence)
        verb_groups[normalized_name(verb)].append(occurrence)

    unique_rows: list[dict[str, Any]] = []
    for records in verb_groups.values():
        example = records[0]
        type_signatures = sorted({row["inferred_argument_types"] for row in records})
        arities = sorted({0 if not row["nl2p_arguments"] else row["nl2p_arguments"].count(" | ") + 1 for row in records})
        unique_rows.append(
            {
                "story": story,
                "nl2p_verb": example["nl2p_verb"],
                "occurrences": len(records),
                "observed_arities": " | ".join(map(str, arities)),
                "inferred_type_signatures": "; ".join(type_signatures),
                "best_pddl_action": example["best_pddl_action"],
                "name_similarity": example["name_similarity"],
            }
        )

    pddl_type_counts = Counter(
        parameter_type for action in domain.actions for parameter_type in action.parameter_types
    )
    strict_nl2p_names = {normalized_name(row["nl2p_verb"]) for row in unique_rows}
    strict_pddl_names = {normalized_name(action.name) for action in domain.actions}
    pddl_covered = {
        row["best_pddl_action"] for row in unique_rows if row["name_similarity"] >= 0.5
    }
    summary = {
        "story": story,
        "nl2p_event_mentions": len(occurrences),
        "nl2p_unique_verbs": len(unique_rows),
        "pddl_action_schemas": len(domain.actions),
        "strict_name_overlap": len(strict_nl2p_names & strict_pddl_names),
        "lexically_linked_nl2p_verbs": sum(row["name_similarity"] >= 0.5 for row in unique_rows),
        "lexically_covered_pddl_schemas": len(pddl_covered),
        "nl2p_arguments": sum(nl2p_type_counts.values()),
        "typed_nl2p_arguments": sum(
            count for object_type, count in nl2p_type_counts.items() if object_type != "unknown"
        ),
        "argument_type_coverage": round(
            1 - nl2p_type_counts["unknown"] / max(1, sum(nl2p_type_counts.values())), 3
        ),
    }
    return summary, occurrences, unique_rows, nl2p_type_counts, pddl_type_counts


def build_report(
    summaries: list[dict[str, Any]], type_rows: list[dict[str, Any]]
) -> str:
    total_events = sum(row["nl2p_event_mentions"] for row in summaries)
    total_unique = sum(row["nl2p_unique_verbs"] for row in summaries)
    total_schemas = sum(row["pddl_action_schemas"] for row in summaries)
    total_strict = sum(row["strict_name_overlap"] for row in summaries)
    total_linked = sum(row["lexically_linked_nl2p_verbs"] for row in summaries)
    total_covered = sum(row["lexically_covered_pddl_schemas"] for row in summaries)
    total_args = sum(row["nl2p_arguments"] for row in summaries)
    typed_args = sum(row["typed_nl2p_arguments"] for row in summaries)
    lines = [
        "# NL2P 与短故事 Domain PDDL 的 Action 比较",
        "",
        "## 总览",
        "",
        f"六个故事中，NL2P 共提取 **{total_events} 条事件 mention**、**{total_unique} 个故事内去重的 verb label**；相应 PDDL 共定义 **{total_schemas} 个 action schema**。",
        f"严格规范化名称重合为 **{total_strict}**。使用简单词形归一化和词元 F1（阈值 0.5）后，**{total_linked}/{total_unique}** 个 NL2P verb 可与某个 PDDL action 建立词汇联系，覆盖 **{total_covered}/{total_schemas}** 个 PDDL schema。",
        f"NL2P 的 {total_args} 个 argument 中，**{typed_args} 个（{typed_args / max(1, total_args):.1%}）**可根据该故事 PDDL constants 的词汇映射推断为 `entity`、`item` 或 `location`。",
        "",
        "| Story | NL2P mentions | Unique verbs | PDDL schemas | Exact names | Linked verbs | Covered schemas | Type coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['story']} | {row['nl2p_event_mentions']} | {row['nl2p_unique_verbs']} | "
            f"{row['pddl_action_schemas']} | {row['strict_name_overlap']} | "
            f"{row['lexically_linked_nl2p_verbs']} | {row['lexically_covered_pddl_schemas']} | "
            f"{row['argument_type_coverage']:.1%} |"
        )
    lines.extend([
        "",
        "## Object/parameter type 分布",
        "",
        "下表的 NL2P 一侧是从 argument 文本映射到 PDDL constants 后得到的粗粒度类型；PDDL 一侧是 action schema 的 parameter 声明。两者分母不同，因此适合比较建模偏向，不应直接解释为准确率。",
        "",
        "| Story | Type | NL2P inferred arguments | PDDL schema parameters |",
        "| --- | --- | ---: | ---: |",
    ])
    for row in type_rows:
        lines.append(
            f"| {row['story']} | {row['object_type']} | {row['nl2p_argument_count']} | {row['pddl_parameter_count']} |"
        )
    lines.extend([
        "",
        "## 如何解读",
        "",
        "- **粒度不同**：NL2P 保留叙事中的逐次事件和重复事件；PDDL 把它们抽象成可复用 schema，并可能把多个连续事件合并为一个 action。",
        "- **命名方式不同**：NL2P 通常给出表层动词短语（如 `fell in`、`boiled`），PDDL 名称常加入施事、受事和结果（如 `wolf_falls_into_pot_and_is_boiled`）。因此 exact overlap 很低是预期现象。",
        "- **参数语义不同**：NL2P arguments 来自自然语言提及，可能省略施事；PDDL parameters 是执行 schema 所需的完整变量。`type_overlap` 只能作为结构相似度提示。",
        "- **类型推断是启发式的**：只有 argument 与 PDDL constant 名称或稳定的名词词元可对应时才赋类型；`unknown` 不代表 NL2P 提取错误。",
        "",
        "详细记录见 `summary_by_story.csv`、`pddl_action_matches.csv`、`nl2p_unique_verbs.csv`、`nl2p_event_matches.csv` 和 `type_distribution.csv`。建议人工复核 `name_similarity >= 0.5` 的候选对应，再据此制作论文中的定性样例表。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nl2p-dir", type=Path, default=DEFAULT_NL2P_DIR)
    parser.add_argument("--pddl-dir", type=Path, default=DEFAULT_PDDL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    all_occurrences: list[dict[str, Any]] = []
    all_unique: list[dict[str, Any]] = []
    pddl_match_rows: list[dict[str, Any]] = []
    type_rows: list[dict[str, Any]] = []

    for prefix, (story, json_name) in STORIES.items():
        payload = json.loads((args.nl2p_dir / json_name).read_text(encoding="utf-8"))
        domain = parse_domain(args.pddl_dir / f"{prefix}_NP_domainfile.pddl")
        summary, occurrences, unique_rows, nl2p_types, pddl_types = analyze_story(
            story, payload, domain
        )
        summaries.append(summary)
        all_occurrences.extend(occurrences)
        all_unique.extend(unique_rows)

        for action in domain.actions:
            candidates = sorted(
                (
                    (lexical_similarity(row["nl2p_verb"], action.name), row)
                    for row in unique_rows
                ),
                key=lambda item: (item[0], item[1]["nl2p_verb"]),
                reverse=True,
            )
            score, best = candidates[0]
            pddl_match_rows.append({
                "story": story,
                "pddl_action": action.name,
                "pddl_parameter_types": " | ".join(action.parameter_types),
                "best_nl2p_verb": best["nl2p_verb"],
                "nl2p_occurrences": best["occurrences"],
                "name_similarity": round(score, 3),
                "lexically_linked": score >= 0.5,
                "nl2p_inferred_type_signatures": best["inferred_type_signatures"],
            })

        for object_type in (*domain.types, "unknown"):
            type_rows.append({
                "story": story,
                "object_type": object_type,
                "nl2p_argument_count": nl2p_types[object_type],
                "pddl_parameter_count": pddl_types[object_type],
            })

    summary_fields = list(summaries[0])
    _write_csv(args.output_dir / "summary_by_story.csv", summaries, summary_fields)
    _write_csv(args.output_dir / "nl2p_event_matches.csv", all_occurrences, list(all_occurrences[0]))
    _write_csv(args.output_dir / "nl2p_unique_verbs.csv", all_unique, list(all_unique[0]))
    _write_csv(args.output_dir / "pddl_action_matches.csv", pddl_match_rows, list(pddl_match_rows[0]))
    _write_csv(args.output_dir / "type_distribution.csv", type_rows, list(type_rows[0]))
    (args.output_dir / "README.md").write_text(
        build_report(summaries, type_rows), encoding="utf-8"
    )
    print(f"Compared {len(summaries)} stories; wrote reports to {args.output_dir}")


if __name__ == "__main__":
    main()
