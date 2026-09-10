# Prepared P1 data contract

Import a JSON object with exactly `domain`, `synthetic`, `provenance`, and `rows`. `domain` is `P1`; `synthetic` must explicitly state whether the slice is constructed. `provenance` contains `source`, `license`, `retrieved_at`, `raw_sha256`, and `availability_basis`. Hashes identify snapshots but are not externally authenticated by this local adapter.

Each row contains:

- `id`, `event_group`, and `split` (`train`, `development`, `confirmation`).
- Original `contract`, a verbatim `source_span`, and `source_id`.
- Curator-verified semantics: `predicate` (`greater_than`), finite numeric `threshold`, `unit` (`F`, `C`, `count`, `USD`, or `percent`), `entity`, `endpoint`, `event_time`, `timezone`.
- `rules_available_at`, `observed_at`, `available_at`, `cutoff`, and `resolved_at`, all ISO timestamps with timezone offsets.
- `price`, a finite 0–1 probability proxy; and `label`, integer 0 or 1 from actual settlement.

Required ordering: observed ≤ first available ≤ cutoff < resolution; rules must be available by cutoff; event_time must follow cutoff. All rows in an event group use one cutoff and one split. Each split contains at least two full groups. All earlier-split labels must be resolved before the next split's earliest cutoff. The adapter admits 6–10,000 rows and the web request envelope is 4 MiB.

Relations require exact matching entity, endpoint, event time, timezone, units and cutoff. Mixed units are excluded from joining, not silently converted. Free text is preserved but not automatically proven equivalent to extracted predicates. The curator must verify this mapping. First availability cannot be inferred from ingestion time.

`synthetic_pack()` in `symplex/fixtures.py` is an executable format example. It is not an empirical archive. Other uploaded CSV/JSON files are stored as unverified context and schema samples until a corresponding executable adapter admits them.
