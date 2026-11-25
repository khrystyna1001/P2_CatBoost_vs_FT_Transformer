# M6 — 5-minute pitch

Deck: [`report/pitch/m6_pitch.pptx`](../report/pitch/m6_pitch.pptx) (5 slides, speaker notes in the file; source
`report/pitch/build_pitch.js`, rebuild with `node report/pitch/build_pitch.js`).

**One key result:** on Amazon Employee Access (a feature with 7 518 levels) CatBoost beats the FT-Transformer by
+0.063 test ROC-AUC (paired-bootstrap 95 % CI [0.047, 0.078]); on adult it is better by 0.012; on bank-marketing the two are
practically equivalent (ROPE ±0.005).

| Slide | Time | Message |
|---|---|---|
| 1. Question and answer | 0:30 | Not on high-cardinality data; the deep model only ties on mixed data. |
| 2. Fair protocol | 0:45 | Same splits and budget, equal tuning, bootstrap with a pre-registered ROPE. |
| 3. Key result | 1:30 | Test AUC chart, +0.063 on Amazon. |
| 4. Why | 1:30 | Seen/unseen levels, SHAP vs attention (τ 0.36–0.61), embeddings learn the target. |
| 5. Takeaways | 0:45 | Prefer CatBoost; serving cost; the 12-trial caveat. |

All numbers come from `docs/m4_results.md` and `docs/m5_deployment.md`. Likely questions: why only 12 trials (CPU budget,
see Threats to validity), why attention ≠ SHAP, what happens with an unseen category in the API (flagged in the response,
mapped to index 0 by the FT-Transformer).
