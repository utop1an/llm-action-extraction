# RISeC, pinned complete release

Source: https://github.com/YiweiJiang2015/RISeC

Commit: `70b9a0b4c4eb656375a83bc1c0ca47c29ec2da2c`.
MIT license: `upstream/LICENSE`.

`source_bundle.json` contains exact UTF-8 file bytes represented as JSON strings,
paths, Git blob SHA-1 values and sizes. `scripts/materialize_risec.py` validates
all hashes before writing `upstream/`. `source_audit.json` records the official
126/56/78 document split and an excluded orphan `.ann` file.

`converted/` preserves original spans and relations under a frozen legacy
conversion policy. The `implicit_arguments` field name is too broad: it holds
both marked implicit/anaphoric links and other action-to-action relations.
No ES/EX/OP labels are supplied or inferred. Current training uses complete BIO
action/argument spans and role-agnostic explicit argument links. The earlier
first-token projection is retained only as a historical experiment and must
not be reported as a complete-name extractor. Implicit arguments and condition clauses are excluded
from the extractive argument task, not relabelled as explicit arguments.

Counts: 260 documents, 3,061 actions and 5,520 extraction-target links under
the legacy policy. Of those links, 45 have the undefined role label `TODO`
(29 train / 6 dev / 10 test); the other 5,475 have `Arg*` labels. Roles are
discarded for this task, but inclusion of unresolved relation labels needs
to be disclosed. Purpose clauses are included; condition clauses are excluded.

The 1,695 excluded links previously described as all "implicit" consist of
1,528 `_X`/`_Z`-marked relations, 111 temporal action relations (`Next` or
`Simultaneous`), and 56 other action-target relations. The latter include 19
`TODO` links. These counts describe the actual release labels, not a manual
resolution of the annotation semantics. The frozen conversion and checkpoints
are retained unchanged so historical results remain reproducible.
All annotation substrings match the source text. This validates conversion
integrity, not the correctness or completeness of every human annotation.

`features_v1/` is a regenerable cache of frozen original GloVe+ELMo representations.
`pipeline_features_seed0/` is built from predicted test action locations, with no
gold action masks or argument labels. Both caches are excluded from Git.

`features_spans_v2/` regenerates full BIO targets while reusing frozen token
embeddings. `pipeline_spans_v2_seed0/` uses only the selected model's predicted
action spans to construct argument contexts; no gold labels enter inference.
These caches are also excluded from Git.

See `docs/dev-notes/2026-11-18-risec-spans-v2.md` for the current protocol and
`results/ceasdrl_risec_20261118/spans_v2/SUMMARY.md` for verified results.
See `results/ceasdrl_risec_20261118/spans_v2/audit_review/AUDIT.md` for the
independent raw-annotation audit and reference-policy sensitivity analysis.
