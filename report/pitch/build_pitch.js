// M6 — 5-minute pitch. Build: node report/pitch/build_pitch.js  (needs pptxgenjs)
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "CatBoost vs. FT-Transformer on categorical data";
pres.author = "Krystyna, RAE sp. z o.o.";

const DARK = "17323A", LIGHT = "FFFFFF", MUTE = "5B6B70", CAT = "1F7A6D", FTT = "D9822B", PALE = "EEF3F2";
const HF = "Cambria", BF = "Calibri";

function title(s, t, color) {
  s.addText(t, { x: 0.6, y: 0.35, w: 8.8, h: 0.9, fontFace: HF, fontSize: 30, bold: true, color: color || DARK, margin: 0, isTextBox: true, valign: "top" });
}
function foot(s, t) {
  s.addText(t, { x: 0.6, y: 5.1, w: 8.8, h: 0.3, fontFace: BF, fontSize: 10, color: MUTE, margin: 0, isTextBox: true });
}

// 1 — title / answer first
let s = pres.addSlide(); s.background = { color: DARK };
s.addText("Does a deep model beat CatBoost on categorical data?", { x: 0.6, y: 0.9, w: 8.8, h: 1.5, fontFace: HF, fontSize: 38, bold: true, color: LIGHT, margin: 0, isTextBox: true, valign: "top" });
s.addText("Not when categories have thousands of levels. CatBoost gains up to +0.063 ROC-AUC; on mixed data the FT-Transformer only draws level.", { x: 0.6, y: 2.7, w: 8.2, h: 1.1, fontFace: BF, fontSize: 20, color: "CFE3DF", margin: 0, isTextBox: true, valign: "top" });
s.addText("Joint capstone P2 · Applied ML × Deep Learning · Krystyna, UEK Kraków", { x: 0.6, y: 4.8, w: 8.8, h: 0.35, fontFace: BF, fontSize: 12, color: "9DB8B3", margin: 0, isTextBox: true });
s.addNotes("Answer first: on three public data sets, with identical splits and an equal tuning budget, CatBoost is never worse than the FT-Transformer and is clearly better when a feature has thousands of levels. The next slides show how I know that is a fair comparison and what the one key number is.");

// 2 — fair protocol
s = pres.addSlide(); s.background = { color: LIGHT };
title(s, "Same data, same budget, one test look");
const cards = [
  ["3 data sets", "adult 14 features · bank-marketing 16 · Amazon access 9 (RESOURCE has 7 518 levels)"],
  ["One split", "60/20/20, seed 2026, preprocessing fitted on train only; leakage checklist passed"],
  ["Equal budget", "12 Optuna trials per model family; 3 seeds classical, 5 seeds deep"],
  ["Real statistics", "paired bootstrap on test rows; 'equivalent' only if the CI sits inside ±0.005 AUC"],
];
cards.forEach((c, i) => {
  const x = 0.6 + (i % 2) * 4.5, y = 1.55 + Math.floor(i / 2) * 1.6;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 4.2, h: 1.4, fill: { color: PALE }, line: { color: PALE }, rectRadius: 0.12, objectName: "card" + i });
  s.addText(c[0], { x: x + 0.25, y: y + 0.15, w: 3.7, h: 0.4, fontFace: HF, fontSize: 18, bold: true, color: CAT, margin: 0, isTextBox: true });
  s.addText(c[1], { x: x + 0.25, y: y + 0.6, w: 3.7, h: 0.75, fontFace: BF, fontSize: 14, color: DARK, margin: 0, isTextBox: true, valign: "top" });
});
foot(s, "Models: one-hot+LogReg, ordinal+HistGB, target-encoding+LightGBM, CatBoost  vs  MLP with embeddings, FT-Transformer.");
s.addNotes("Every difference you see afterwards comes from the model, not the experiment. Splits fixed once, everything that learns is fitted on train, the same Optuna budget for both families. The budget is small, 12 trials, because I ran on two CPU cores; I come back to that at the end.");

// 3 — the key result: chart
s = pres.addSlide(); s.background = { color: LIGHT };
title(s, "CatBoost wins where categories are many");
s.addImage({ path: __dirname + "/pitch_test_auc.png", x: 0.5, y: 1.2, w: 5.8, h: 3.7, altText: "pitch_test_auc.png" });
s.addText("+0.063", { x: 6.5, y: 1.45, w: 3.0, h: 1.0, fontFace: HF, fontSize: 60, bold: true, color: CAT, margin: 0, isTextBox: true });
s.addText("ROC-AUC on Amazon, 95% CI [0.047, 0.078]", { x: 6.5, y: 2.45, w: 3.0, h: 0.7, fontFace: BF, fontSize: 14, color: DARK, margin: 0, isTextBox: true, valign: "top" });
s.addText("adult +0.012, clearly better. bank-marketing +0.002: practically equivalent (inside ±0.005).", { x: 6.5, y: 3.35, w: 3.0, h: 1.2, fontFace: BF, fontSize: 14, color: DARK, margin: 0, isTextBox: true, valign: "top" });
foot(s, "Paired bootstrap, 2 000 resamples; difference = CatBoost minus FT-Transformer. Axis starts at 0.75.");
s.addNotes("This is the key result. Amazon has nine categorical features and one of them has 7 518 levels. CatBoost scores 0.870, the FT-Transformer 0.807: a 0.063 gap whose bootstrap interval is far from zero. On adult CatBoost is also better, by 0.012. Only on bank-marketing are the two statistically and practically the same. Note the y-axis starts at 0.75 to make the gaps readable.");

// 4 — why: unseen + explainability
s = pres.addSlide(); s.background = { color: LIGHT };
title(s, "Why: unseen levels and explanations");
s.addImage({ path: __dirname + "/pitch_seen_unseen.png", x: 0.5, y: 1.2, w: 4.7, h: 3.76, altText: "pitch_seen_unseen.png" });
const pts = [
  ["Unseen levels hurt both", "CatBoost drops 0.058, the FT-Transformer 0.042; CatBoost still stays ahead (0.825 vs 0.780)."],
  ["Attention is not SHAP", "Kendall τ between CatBoost SHAP and FT-T attention: 0.36 / 0.45 / 0.61; attention ranks are unstable across seeds."],
  ["Embeddings learn the target", "PC1 of the 305 RESOURCE embeddings correlates 0.81 with the level's target mean."],
];
pts.forEach((p, i) => {
  const y = 1.35 + i * 1.2;
  s.addText(p[0], { x: 5.4, y, w: 4.1, h: 0.35, fontFace: HF, fontSize: 16, bold: true, color: CAT, margin: 0, isTextBox: true });
  s.addText(p[1], { x: 5.4, y: y + 0.38, w: 4.1, h: 0.75, fontFace: BF, fontSize: 13, color: DARK, margin: 0, isTextBox: true, valign: "top" });
});
foot(s, "Unseen = 1 087 of 6 554 Amazon test rows. FT-Transformer maps rare/unseen levels to a reserved embedding index 0.");
s.addNotes("Why does CatBoost win? My hypothesis was that it would degrade less on unseen categories; that was wrong: it loses more, 0.058 versus 0.042, but it starts higher and stays ahead. For explainability, SHAP and attention only moderately agree, tau 0.36 to 0.61, and attention is unstable across seeds, so attention should not be read as an explanation. A positive finding: the FT-Transformer embeddings learn a direction that correlates 0.81 with the target mean, without ever seeing a target statistic.");

// 5 — cost + takeaways
s = pres.addSlide(); s.background = { color: DARK };
title(s, "Takeaways", LIGHT);
const tk = [
  ["Default to CatBoost", "for high-cardinality tables: better, up to 3× lower HTTP latency at batch 1 000, no GPU needed."],
  ["FT-Transformer is competitive", "on mixed data (bank-marketing) but costs 10–20× more model latency than CatBoost."],
  ["Served both", "FastAPI + ONNX Runtime: p95 ≤ 13 ms for a single record; Dockerfile provided."],
  ["Caveat", "12 Optuna trials, 2 CPU cores: the tuned FT-Transformer is below its own M3 baseline on Amazon, so the gap is probably overstated."],
];
tk.forEach((p, i) => {
  const y = 1.35 + i * 0.95;
  s.addText(p[0], { x: 0.6, y, w: 3.2, h: 0.8, fontFace: HF, fontSize: 18, bold: true, color: FTT, margin: 0, isTextBox: true, valign: "top" });
  s.addText(p[1], { x: 3.9, y, w: 5.5, h: 0.8, fontFace: BF, fontSize: 15, color: "E6F0EE", margin: 0, isTextBox: true, valign: "top" });
});
s.addNotes("Close with the practical recommendation and the honest caveat. The model-only latency of CatBoost is 0.2-0.3 ms versus 2.6-3.5 ms for the ONNX or PyTorch FT-Transformer at batch 1; through HTTP both answer a single record in under 13 ms at p95. The limitation I want you to remember: with only 12 tuning trials on CPU the FT-Transformer is under-tuned, so +0.063 is an upper bound on the true gap. Questions?");

pres.writeFile({ fileName: __dirname + "/m6_pitch.pptx" }).then(() => console.log("written"));
