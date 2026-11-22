"""Lossless RISeC span targets and text-only inference contexts (Python 3.6+).

BIO distinguishes adjacent mentions. No role, ES/EX/OP, implicit argument,
gold-span expansion, or vocabulary-based name completion is used.
"""
import hashlib
import json


def embedding_key(tokens):
    return hashlib.sha256(json.dumps(tokens, ensure_ascii=False).encode('utf-8')).hexdigest()


def token_span(indices):
    indices = sorted(set(indices))
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError('Empty or discontinuous span cannot be encoded by BIO')
    return [indices[0], indices[-1] + 1]


def encode_bio(length, spans):
    labels = [0] * length
    for start, end in sorted(set(tuple(s) for s in spans)):
        if not 0 <= start < end <= length:
            raise ValueError('Span outside context')
        if any(labels[start:end]):
            raise ValueError('Overlapping spans cannot be encoded by BIO')
        labels[start:end] = [1] + [2] * (end - start - 1)
    return labels


def decode_bio(labels):
    """O=0, B=1, I=2; an I after O is deterministically treated as B."""
    spans = []
    for i, label in enumerate(labels):
        if label not in (0, 1, 2):
            raise ValueError('Unknown BIO label')
        if label == 1 or (label == 2 and (i == 0 or labels[i - 1] == 0)):
            spans.append([i, i + 1])
        elif label == 2:
            spans[-1][1] = i + 1
    return spans


def context(words, action_span, capacity=100, require_complete_action=True):
    if not 0 <= action_span[0] < action_span[1] <= len(words):
        raise ValueError('Action span outside document')
    start = max(0, min(action_span[0] - capacity // 2, len(words) - capacity))
    stop = min(len(words), start + capacity)
    if require_complete_action and not start <= action_span[0] < action_span[1] <= stop:
        raise ValueError('Action span exceeds argument context')
    return start, stop


def make_span_examples(records, mode):
    examples = []
    for row in records:
        words = row['words']
        if mode == 'act':
            if len(words) > 600:
                raise ValueError('Document exceeds action capacity')
            spans = sorted(token_span(a['trigger_tokens']) for a in row['actions'])
            examples.append({'id': row['id'], 'doc_id': row['id'], 'tokens': words,
                             'gold_spans': spans, 'labels': encode_bio(len(words), spans)})
        elif mode == 'arg':
            if not row['argument_links_available']:
                raise ValueError('Missing annotated action-argument relations')
            for action in row['actions']:
                action_span = token_span(action['trigger_tokens'])
                start, stop = context(words, action_span)
                spans = sorted(set(tuple(token_span(a['token_indices'])) for a in action['explicit_arguments']))
                spans = [[a - start, b - start] for a, b in spans]
                # Fail instead of silently discarding an out-of-window argument.
                labels = encode_bio(stop - start, spans)
                examples.append({'id': '{}:{}'.format(row['id'], action_span[0]), 'doc_id': row['id'],
                                 'tokens': words[start:stop], 'gold_spans': spans, 'labels': labels,
                                 'act_ind': action_span[0] - start, 'action_span': action_span,
                                 'window_start': start})
        else:
            raise ValueError('Unknown extraction mode')
    for ex in examples:
        ex['embedding_key'] = embedding_key(ex['tokens'])
        if decode_bio(ex['labels']) != ex['gold_spans']:
            raise ValueError('BIO round-trip loses annotations')
    return examples


def predicted_span_contexts(records, predictions):
    """Reads only words and predicted spans. Gold annotations are never accessed."""
    doc_ids = [p['doc_id'] for p in predictions]
    if len(doc_ids) != len(set(doc_ids)) or set(records) != set(doc_ids):
        raise ValueError('Prediction document set mismatch')
    examples = []
    for prediction in predictions:
        doc_id = prediction['doc_id']
        words = records[doc_id]['words']
        spans = prediction['predicted_spans']
        encode_bio(len(words), spans)  # Validate all spans before using them.
        if len(spans) != len(set(tuple(s) for s in spans)):
            raise ValueError('Duplicate predicted actions')
        for span in spans:
            # A model can incorrectly emit an overlong action. Keep it for
            # false-positive accounting instead of crashing or dropping it.
            start, stop = context(words, span, require_complete_action=False)
            tokens = words[start:stop]
            examples.append({'id': '{}:pred{}'.format(doc_id, span[0]), 'doc_id': doc_id,
                             'tokens': tokens, 'embedding_key': embedding_key(tokens),
                             'act_ind': span[0] - start, 'action_span': span,
                             'window_start': start})
    return examples


def span_name(record, span):
    start, end = span
    if not 0 <= start < end <= len(record['words']):
        raise ValueError('Name span outside document')
    return record['text'][record['token_offsets'][start][0]:record['token_offsets'][end - 1][1]]


def count_scores(right, predicted, gold):
    return {'precision': right / predicted if predicted else 0.,
            'recall': right / gold if gold else 0.,
            'f1': 2. * right / (predicted + gold) if predicted + gold else 0.,
            'right': right, 'predicted': predicted, 'gold': gold}


def span_scores(predictions, examples):
    if len(predictions) != len(examples):
        raise ValueError('Prediction/example count mismatch')
    right = predicted = gold = 0
    output = []
    for labels, ex in zip(predictions, examples):
        if len(labels) != len(ex['tokens']):
            raise ValueError('Predicted label/token count mismatch')
        spans = decode_bio(labels)
        found = set(tuple(s) for s in spans)
        expected = set(tuple(s) for s in ex.get('gold_spans', []))
        right += len(found & expected)
        predicted += len(found)
        gold += len(expected)
        output.append({'id': ex['id'], 'doc_id': ex['doc_id'], 'predicted_spans': spans,
                       'predicted_labels': labels})
    return count_scores(right, predicted, gold), output
