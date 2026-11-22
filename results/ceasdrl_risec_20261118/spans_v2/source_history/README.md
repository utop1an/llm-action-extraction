# Source history

Frozen files in this directory match historical source hashes from the completed development runs. The original `risec_spans.py` predates input-length guards and handling of overlong predicted actions; these guards do not change canonical targets or scores for valid predictions. The earlier trainer and model helper predate the separately recorded balanced-replay / mention-rate candidate. Stage receipt source hashes are captured at completion. Final source checks accept a current file or an exact matching frozen version, without rewriting historical receipts.
