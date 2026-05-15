"use strict";

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { imageSizingContain } = require("./pptxgenjs_helpers/image");
const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./pptxgenjs_helpers/layout");

const OUT_DIR = __dirname;
const ASSET_DIR = path.join(OUT_DIR, "assets");
const metrics = JSON.parse(
  fs.readFileSync(path.join(OUT_DIR, "metrics.json"), "utf8")
);

const palette = {
  deepGreen: "0B6E4F",
  burntOrange: "C84C09",
  slate: "2F4858",
  sand: "EAD8B1",
  mist: "F4F1EA",
  warmWhite: "FBFAF7",
  ink: "1E2A33",
  brown: "7A5C3E",
};

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "OpenAI Codex";
pptx.company = "PrOMMiS";
pptx.subject = "PC-SAFT full integration tutorial summary";
pptx.title = "PC-SAFT Integration Tutorial Slides";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "en-US",
};

function addBackground(slide, accent = palette.deepGreen) {
  slide.background = { color: palette.warmWhite };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: 13.333,
    h: 0.3,
    line: { color: accent, transparency: 100 },
    fill: { color: accent, transparency: 0 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 7.18,
    w: 13.333,
    h: 0.32,
    line: { color: palette.slate, transparency: 100 },
    fill: { color: palette.slate, transparency: 0 },
  });
}

function addTitle(slide, title, subtitle) {
  slide.addText(title, {
    x: 0.55,
    y: 0.5,
    w: 7.8,
    h: 0.5,
    fontFace: "Aptos Display",
    fontSize: 24,
    bold: true,
    color: palette.ink,
    margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.57,
      y: 1.08,
      w: 9.0,
      h: 0.35,
      fontFace: "Aptos",
      fontSize: 10.5,
      color: palette.slate,
      margin: 0,
    });
  }
}

function addFooter(slide, text) {
  slide.addText(text, {
    x: 0.55,
    y: 7.23,
    w: 11.9,
    h: 0.14,
    fontFace: "Aptos",
    fontSize: 8.5,
    color: "FFFFFF",
    margin: 0,
    align: "left",
  });
}

function addBulletList(slide, items, box, color = palette.ink) {
  const runs = [];
  items.forEach((item) => {
    runs.push({
      text: item,
      options: {
        bullet: { indent: 14 },
        breakLine: true,
      },
    });
  });
  slide.addText(runs, {
    ...box,
    fontFace: "Aptos",
    fontSize: 15,
    color,
    margin: 0,
    breakLine: false,
    paraSpaceAfterPt: 10,
    valign: "top",
  });
}

function addInfoCard(slide, title, body, box, fillColor) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x: box.x,
    y: box.y,
    w: box.w,
    h: box.h,
    rectRadius: 0.08,
    line: { color: fillColor, width: 1.1 },
    fill: { color: palette.mist },
  });
  slide.addText(title, {
    x: box.x + 0.18,
    y: box.y + 0.12,
    w: box.w - 0.36,
    h: 0.3,
    fontFace: "Aptos",
    fontSize: 12.5,
    bold: true,
    color: fillColor,
    margin: 0,
  });
  slide.addText(body, {
    x: box.x + 0.18,
    y: box.y + 0.44,
    w: box.w - 0.36,
    h: box.h - 0.58,
    fontFace: "Aptos",
    fontSize: 11,
    color: palette.ink,
    margin: 0,
    valign: "top",
  });
}

function addMetricCard(slide, label, value, note, box, accent) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x: box.x,
    y: box.y,
    w: box.w,
    h: box.h,
    rectRadius: 0.06,
    line: { color: accent, transparency: 100 },
    fill: { color: accent, transparency: 0 },
  });
  slide.addText(label, {
    x: box.x + 0.16,
    y: box.y + 0.14,
    w: box.w - 0.32,
    h: 0.22,
    fontFace: "Aptos",
    fontSize: 11,
    color: "FFFFFF",
    bold: true,
    margin: 0,
  });
  slide.addText(value, {
    x: box.x + 0.16,
    y: box.y + 0.45,
    w: box.w - 0.32,
    h: 0.38,
    fontFace: "Aptos Display",
    fontSize: 20,
    color: "FFFFFF",
    bold: true,
    margin: 0,
  });
  slide.addText(note, {
    x: box.x + 0.16,
    y: box.y + 0.9,
    w: box.w - 0.32,
    h: 0.24,
    fontFace: "Aptos",
    fontSize: 9.5,
    color: "F7F5F0",
    margin: 0,
  });
}

function imagePath(name) {
  return path.join(ASSET_DIR, name);
}

function fmt(value, digits = 3) {
  return Number(value).toFixed(digits);
}

function buildDeck() {
  const maxDiff = Math.max(...metrics.comparison.map((row) => row["Absolute difference"]));
  const oaFirst = metrics.sensitivity[0];
  const oaLast = metrics.sensitivity[metrics.sensitivity.length - 1];

  let slide = pptx.addSlide();
  addBackground(slide, palette.deepGreen);
  addTitle(
    slide,
    "PC-SAFT to PrOMMiS Integration Flow",
    "Summary slides from the full tutorial notebook for chemical engineers learning the detailed-chemistry-to-process-model workflow."
  );
  addBulletList(
    slide,
    [
      "Direct PC-SAFT supplies the single-stage liquid-liquid equilibrium truth model.",
      "pcsaft-pse converts that chemistry output into process-facing quantities such as extraction efficiency and distribution ratio.",
      "PrOMMiS embeds those quantities in an IDAES MSContactor stage model so outlet streams and KPIs can be solved at the process level.",
    ],
    { x: 0.65, y: 1.45, w: 4.25, h: 2.45 }
  );
  addInfoCard(
    slide,
    "Case study basis",
    `Aqueous ions: Li, Na, Cl\nOrganic phase: TOP + Hexane\nUnit operation: one equilibrium contact stage\nNominal O/A ratio: ${fmt(metrics.feed_basis.o_to_a_ratio, 2)}`,
    { x: 0.68, y: 4.15, w: 3.95, h: 1.6 },
    palette.burntOrange
  );
  slide.addImage({
    path: imagePath("workflow_overview.png"),
    ...imageSizingContain(imagePath("workflow_overview.png"), 5.2, 1.45, 7.4, 4.75),
  });
  slide.addText(
    "Use this pattern when you want thermodynamic fidelity where it matters but still need a flowsheet-friendly unit model for design studies and scale-up.",
    {
      x: 5.28,
      y: 6.05,
      w: 7.0,
      h: 0.7,
      fontFace: "Aptos",
      fontSize: 12,
      color: palette.slate,
      margin: 0,
    }
  );
  addFooter(slide, "Tutorial source: docs/tutorials/pcsaft_full_integration.ipynb");
  warnIfSlideHasOverlaps(slide, pptx);
  warnIfSlideElementsOutOfBounds(slide, pptx);

  slide = pptx.addSlide();
  addBackground(slide, palette.burntOrange);
  addTitle(
    slide,
    "How the Coupling Actually Works",
    "The tutorial is valuable because it shows the specific handoff variables and constraints rather than treating the surrogate as a black box."
  );
  addInfoCard(
    slide,
    "Step 1",
    "Define a chemically consistent feed basis. Chloride is back-calculated from electroneutrality so the aqueous chemistry closes cleanly.",
    { x: 0.65, y: 1.45, w: 2.9, h: 1.45 },
    palette.slate
  );
  addInfoCard(
    slide,
    "Step 2",
    "Run the direct PC-SAFT stage calculation to obtain lithium extraction efficiency, lithium distribution ratio, and calibration values.",
    { x: 3.78, y: 1.45, w: 2.9, h: 1.45 },
    palette.burntOrange
  );
  addInfoCard(
    slide,
    "Step 3",
    "Build the PrOMMiS one-stage solvent extraction model and pass the detailed-chemistry outputs into process-level variables.",
    { x: 6.91, y: 1.45, w: 2.9, h: 1.45 },
    palette.deepGreen
  );
  addInfoCard(
    slide,
    "Step 4",
    "Replace the default Li mass-transfer relation with a flowsheet-level coupling constraint so the process model follows the surrogate-derived Li behavior.",
    { x: 10.04, y: 1.45, w: 2.6, h: 1.45 },
    palette.brown
  );
  addInfoCard(
    slide,
    "Key chemistry relations",
    "$n_{Cl^-} = n_{Li^+} + n_{Na^+}$\n$K_{Li} = \\eta_{Li}/100$",
    { x: 0.72, y: 3.35, w: 3.55, h: 1.5 },
    palette.deepGreen
  );
  addBulletList(
    slide,
    [
      "Sodium and chloride use direct calibration values in this proof of concept, while lithium partitioning is explicitly tied to the surrogate output.",
      "The integrated model is still a standard solvable PrOMMiS unit, which means it can sit inside larger flowsheets and optimization studies.",
      "This is the core integration lesson: preserve the process-relevant chemistry signal while moving into a process-systems framework.",
    ],
    { x: 4.55, y: 3.3, w: 7.5, h: 2.45 }
  );

  addFooter(slide, "Notebook section focus: feed definition, detailed chemistry truth model, and Li transfer coupling");
  warnIfSlideHasOverlaps(slide, pptx);
  warnIfSlideElementsOutOfBounds(slide, pptx);

  slide = pptx.addSlide();
  addBackground(slide, palette.deepGreen);
  addTitle(
    slide,
    "The Integrated Stage Preserves the Chemistry Signal",
    "The point of the bridge is not just to solve; it is to reproduce the detailed-chemistry quantities that drive extraction performance."
  );
  addMetricCard(
    slide,
    "Li extraction",
    `${fmt(metrics.summary.eta_li_pct, 2)}%`,
    "Integrated one-stage result",
    { x: 0.7, y: 1.45, w: 2.35, h: 1.35 },
    palette.deepGreen
  );
  addMetricCard(
    slide,
    "Li distribution ratio",
    fmt(metrics.summary.d_li, 4),
    "Process-level D_Li",
    { x: 0.7, y: 2.95, w: 2.35, h: 1.35 },
    palette.slate
  );
  addMetricCard(
    slide,
    "Aq. outlet Li",
    `${fmt(metrics.summary.li_aq_out_mg_L, 2)} mg/L`,
    "Stage outlet concentration",
    { x: 0.7, y: 4.45, w: 2.35, h: 1.35 },
    palette.burntOrange
  );
  slide.addText(
    `Direct PC-SAFT and the integrated PrOMMiS model agree to within ${maxDiff.toExponential(2)} for the transferred Li metrics. That is the validation step that makes the integration credible.`,
    {
      x: 0.72,
      y: 6.0,
      w: 2.6,
      h: 0.6,
      fontFace: "Aptos",
      fontSize: 11,
      color: palette.ink,
      margin: 0,
    }
  );
  slide.addImage({
    path: imagePath("agreement.png"),
    ...imageSizingContain(imagePath("agreement.png"), 3.45, 1.35, 4.05, 2.45),
  });
  slide.addImage({
    path: imagePath("stage_outcome.png"),
    ...imageSizingContain(imagePath("stage_outcome.png"), 7.75, 1.35, 5.0, 5.35),
  });
  slide.addText(
    "The stage-outcome plot shows what a process engineer needs next: how Li and Na move between inlet and outlet streams once the chemistry layer is embedded in the unit model.",
    {
      x: 3.5,
      y: 4.15,
      w: 3.8,
      h: 1.35,
      fontFace: "Aptos",
      fontSize: 11.5,
      color: palette.slate,
      margin: 0,
    }
  );
  addFooter(slide, "Validation emphasis: transferred Li metrics remain consistent across the chemistry and process layers");
  warnIfSlideHasOverlaps(slide, pptx);
  warnIfSlideElementsOutOfBounds(slide, pptx);

  slide = pptx.addSlide();
  addBackground(slide, palette.burntOrange);
  addTitle(
    slide,
    "Why This Workflow Is Useful",
    "For process engineers, the benefit is not a prettier chemistry calculation. The benefit is a tractable process model informed by thermodynamics."
  );
  slide.addImage({
    path: imagePath("sensitivity.png"),
    ...imageSizingContain(imagePath("sensitivity.png"), 6.0, 1.5, 6.15, 4.95),
  });
  addBulletList(
    slide,
    [
      "You can explore design levers such as organic-to-aqueous ratio while keeping the chemistry basis explicit.",
      `In the tutorial case, increasing O/A from ${fmt(oaFirst["O/A ratio"], 2)} to ${fmt(oaLast["O/A ratio"], 2)} raises Li extraction from ${fmt(oaFirst["Li extraction (%)"], 2)}% to ${fmt(oaLast["Li extraction (%)"], 2)}%.`, 
      "The same pattern can support larger flowsheets, stagewise studies, optimization, and eventually richer fitted surrogates over a wider operating domain.",
      "That makes this a practical bridge from detailed equilibrium science to process synthesis and decision making.",
    ],
    { x: 0.72, y: 1.62, w: 4.7, h: 3.85 }
  );
  addInfoCard(
    slide,
    "Best takeaway for learners",
    "Use direct chemistry to establish the truth model, then expose only the quantities the unit model actually needs. That keeps the process model fast without losing physical meaning.",
    { x: 0.76, y: 5.65, w: 11.5, h: 0.88 },
    palette.deepGreen
  );
  addFooter(slide, "Forward path from the notebook: replace the constant bridge with a fitted surrogate over a broader operating window");
  warnIfSlideHasOverlaps(slide, pptx);
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

async function main() {
  buildDeck();
  const outPath = path.join(OUT_DIR, "pcsaft_full_integration_summary.pptx");
  await pptx.writeFile({ fileName: outPath });
  console.log(`Wrote ${outPath}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});


