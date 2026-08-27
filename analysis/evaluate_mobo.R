#!/usr/bin/env Rscript
# =============================================================================
# Evaluation of the per-participant MOBO runs.
#
#   Rscript analysis/evaluate_mobo.R [path/to/trials.csv]
#
# Input: the CSV produced by analysis/export_firestore.py — one row per
# completed trial, with the five stimulus parameters, the two objective scores,
# the optimizer's phase label, and the hypervolume the service wrote back.
#
# Output: analysis/output/ (figures + tables) and an APA-style console summary.
#
# What it answers
#   1. Does the hypervolume actually improve, per participant and overall?
#   2. Does the MOBO phase beat the Sobol phase, or is the GP earning nothing?
#   3. Which configurations end up on each participant's Pareto front?
#   4. How far is each participant's front from the pooled reference front?
#
# Built on colleyRstats (APA reporting, MOBO plots) and moocore (Pareto/HV/IGD).
# =============================================================================

suppressPackageStartupMessages({
  library(colleyRstats)
  library(moocore)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
})

# ── Configuration ────────────────────────────────────────────────────────────
# MUST stay in sync with the service:
#   REF_POINT   <- optimizer_core.py  (REF_POINT = [-0.1] * n_objectives)
#   N_SOBOL     <- main.py / GET /health
#   OBJECTIVES  <- space.py OBJECTIVE_FIELDS (both MAXIMISED, in [0, 1])
OBJECTIVES <- c("subjectiveScore", "objectiveScore")
REF_POINT  <- c(-0.1, -0.1)
N_SOBOL    <- 12
PARAMS     <- c("intensity", "sharpness", "duration", "interval", "pattern")

# Parameters that `space.canonicalize()` pins to a sentinel when
# pattern == "constant" — they are NOT measurements for those rows and must be
# excluded from any "which parameter mattered" analysis restricted to constant.
DEAD_FOR_CONSTANT <- c("duration", "interval")

args    <- commandArgs(trailingOnly = TRUE)
csv_in  <- if (length(args) >= 1) args[1] else file.path("analysis", "trials.csv")
out_dir <- file.path("analysis", "output")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

colleyRstats_setup(set_theme = TRUE, print_citation = FALSE, verbose = FALSE)

cat("\n=============================================================\n")
cat("MOBO evaluation\n")
cat("=============================================================\n")
cat("input      :", csv_in, "\n")
cat("objectives :", paste(OBJECTIVES, collapse = ", "), "(both maximised)\n")
cat("reference  :", paste(REF_POINT, collapse = ", "), "\n\n")

# ── 1. Load, validate, clean ─────────────────────────────────────────────────
stopifnot(file.exists(csv_in))
raw <- read.csv(csv_in, stringsAsFactors = FALSE)

needed <- c("pid", "phaseStep", "phase", OBJECTIVES)
missing_cols <- setdiff(needed, names(raw))
if (length(missing_cols)) {
  stop("input CSV is missing required columns: ", paste(missing_cols, collapse = ", "))
}

trials <- raw |>
  mutate(
    pid       = as.character(pid),
    phaseStep = as.integer(phaseStep),
    across(all_of(intersect(OBJECTIVES, names(raw))), as.numeric),
    attentionCheckPassed = if ("attentionCheckPassed" %in% names(raw)) {
      !tolower(as.character(attentionCheckPassed)) %in% c("false", "0", "no")
    } else TRUE
  )

n_failed <- sum(!trials$attentionCheckPassed)
trials <- trials |> filter(attentionCheckPassed)
if (n_failed > 0) {
  cat("dropped", n_failed, "trial(s) with a failed attention check",
      "(the service excludes them from training too)\n")
}

out_of_range <- trials |> filter(if_any(all_of(OBJECTIVES), ~ .x < 0 | .x > 1))
if (nrow(out_of_range)) {
  warning("objective scores outside [0, 1] in ", nrow(out_of_range),
          " row(s) — the reference point assumes [0, 1].")
}

# `plot_mobo2()` matches the phase column case-insensitively against exactly
# "sampling" and "optimization" and errors if either is absent. The service
# writes "exploration" / "exploration-fallback" / "optimization", so remap.
# Trials with no phase label (missing parameterValues doc) fall back to the
# N_SOBOL cutoff.
trials <- trials |>
  mutate(
    Phase = case_when(
      grepl("^optimi", phase, ignore.case = TRUE) ~ "Optimization",
      grepl("^explor", phase, ignore.case = TRUE) ~ "Sampling",
      phaseStep > N_SOBOL                         ~ "Optimization",
      TRUE                                        ~ "Sampling"
    ),
    Phase = factor(Phase, levels = c("Sampling", "Optimization"))
  ) |>
  arrange(pid, phaseStep)

participants <- sort(unique(trials$pid))
cat("participants:", length(participants), " trials:", nrow(trials), "\n")
print(trials |> count(pid, Phase) |> tidyr::pivot_wider(names_from = Phase, values_from = n, values_fill = 0))
cat("\n")

# ── 2. Anytime hypervolume, recomputed with moocore ──────────────────────────
# moocore minimises by default; both objectives here are maximised, so
# `maximise = TRUE` everywhere. Getting this wrong silently inverts the front.
anytime_hv <- function(m, reference) {
  vapply(seq_len(nrow(m)),
         function(t) moocore::hypervolume(m[seq_len(t), , drop = FALSE],
                                          reference = reference, maximise = TRUE),
         numeric(1))
}

trials <- trials |>
  group_by(pid) |>
  mutate(hv_r = anytime_hv(as.matrix(pick(all_of(OBJECTIVES))), REF_POINT)) |>
  ungroup()

# monotonicity is a property of the hypervolume indicator — violation means the
# data are out of trial order, not that the optimizer regressed.
non_mono <- trials |> group_by(pid) |>
  summarise(bad = any(diff(hv_r) < -1e-9), .groups = "drop") |> filter(bad)
if (nrow(non_mono)) {
  warning("hypervolume decreased for: ", paste(non_mono$pid, collapse = ", "),
          " — check the trial ordering in the input.")
}

# ── 3. Cross-check against the hypervolume the SERVICE wrote back ────────────
if ("hypervolume" %in% names(trials)) {
  chk <- trials |>
    mutate(hv_service = suppressWarnings(as.numeric(hypervolume))) |>
    filter(!is.na(hv_service)) |>
    mutate(delta = abs(hv_service - hv_r))
  if (nrow(chk)) {
    cat("hypervolume cross-check (service vs moocore):",
        nrow(chk), "trials, max |delta| =", format(max(chk$delta), digits = 3), "\n")
    if (max(chk$delta) > 1e-6) {
      cat("  ^ MISMATCH: the service and this script disagree. Check that\n",
          "    REF_POINT and OBJECTIVES above match optimizer_core.py / space.py.\n")
    } else {
      cat("  OK — botorch and moocore agree to within 1e-6.\n")
    }
  }
  cat("\n")
}

# ── 4. Pareto front per participant ──────────────────────────────────────────
# add_pareto_moocore_column() calls moocore::is_nondominated() with its default
# maximise = FALSE, so feed it NEGATED objectives to get a maximisation front.
neg_names <- paste0("neg_", OBJECTIVES)
trials <- trials |> mutate(across(all_of(OBJECTIVES), ~ -.x, .names = "neg_{.col}"))

trials <- trials |>
  group_by(pid) |>
  group_modify(~ colleyRstats::add_pareto_moocore_column(as.data.frame(.x), neg_names)) |>
  ungroup() |>
  rename(on_pareto = PARETO_MOOCORE)

pareto_summary <- trials |>
  group_by(pid) |>
  summarise(
    trials       = n(),
    pareto_size  = sum(on_pareto),
    final_hv     = dplyr::last(hv_r),
    hv_at_sobol  = dplyr::last(hv_r[Phase == "Sampling"]),
    .groups = "drop"
  ) |>
  mutate(hv_gain_optim = final_hv - hv_at_sobol)

cat("per-participant summary\n")
print(as.data.frame(pareto_summary), row.names = FALSE)
cat("\n")
write.csv(pareto_summary, file.path(out_dir, "participant_summary.csv"), row.names = FALSE)

# ── 5. Did the MOBO phase earn its trials? ───────────────────────────────────
# Hypervolume delta contributed by each phase, divided by that phase's trial
# count — the phases have different budgets (N_SOBOL vs the rest), so the raw
# deltas are not comparable.
#
# ⚠ READ BEFORE REPORTING THIS TEST. The comparison is structurally biased in
# favour of Sampling: it starts from an empty front (HV = 0) and so is credited
# with everything it builds, while Optimization inherits an already-good front
# and can only add under diminishing returns. A null result here is therefore
# NOT evidence that the GP is useless — near-parity already means the MOBO
# trials keep pace with trials that had the whole space to themselves. Treat
# `hv_gain_optim` in the participant summary (the absolute improvement the GP
# added on top of Sobol) as the headline number, and this test as descriptive.
phase_gain <- trials |>
  group_by(pid) |>
  summarise(
    hv_start   = 0,
    hv_sobol   = dplyr::last(hv_r[Phase == "Sampling"]),
    hv_final   = dplyr::last(hv_r),
    n_sobol    = sum(Phase == "Sampling"),
    n_optim    = sum(Phase == "Optimization"),
    .groups = "drop"
  ) |>
  mutate(
    Sampling     = (hv_sobol - hv_start) / pmax(n_sobol, 1),
    Optimization = (hv_final - hv_sobol) / pmax(n_optim, 1)
  ) |>
  select(pid, Sampling, Optimization) |>
  tidyr::pivot_longer(c(Sampling, Optimization),
                      names_to = "Phase", values_to = "hv_gain_per_trial") |>
  mutate(Phase = factor(Phase, levels = c("Sampling", "Optimization")))

cat("hypervolume gain per trial, by phase\n")
print(as.data.frame(phase_gain), row.names = FALSE)
cat("\n")
write.csv(phase_gain, file.path(out_dir, "phase_gain.csv"), row.names = FALSE)

if (length(participants) >= 3 && all(table(phase_gain$Phase) == length(participants))) {
  cat("--- APA report: HV gain per trial, Sampling vs Optimization ---\n")
  try(print(colleyRstats::report_mean_sd(as.data.frame(phase_gain),
                                         iv = "Phase", dv = "hv_gain_per_trial")),
      silent = TRUE)
  p_phase <- try(colleyRstats::plot_within_stats(
    data = as.data.frame(phase_gain), x = "Phase", y = "hv_gain_per_trial",
    ylab = "HV gain per trial"), silent = TRUE)
  if (!inherits(p_phase, "try-error")) {
    try(colleyRstats::save_paper_figure(p_phase,
        filename = file.path(out_dir, "phase_gain.pdf"), columns = 1), silent = TRUE)
    try(invisible(colleyRstats::report_ggstatsplot(p_phase, iv = "Phase",
        dv = "HV gain per trial")), silent = TRUE)
  }
  cat("\n")
} else {
  cat("(skipping the inferential phase comparison: needs >= 3 participants",
      "present in both phases; have", length(participants), ")\n\n")
}

# ── 6. Distance to the pooled reference front ────────────────────────────────
# The true Pareto front is unknown, so the standard substitute is the union of
# all participants' non-dominated points. IGD+ and additive epsilon are then
# comparable across participants (both: lower is better).
ref_front <- moocore::filter_dominated(
  as.matrix(trials[, OBJECTIVES]), maximise = TRUE)
cat("pooled reference front:", nrow(ref_front), "non-dominated points\n")

quality <- trials |>
  group_by(pid) |>
  summarise(
    igd_plus = moocore::igd_plus(as.matrix(pick(all_of(OBJECTIVES))),
                                 reference = ref_front, maximise = TRUE),
    eps_add  = moocore::epsilon_additive(as.matrix(pick(all_of(OBJECTIVES))),
                                         reference = ref_front, maximise = TRUE),
    .groups = "drop"
  )
print(as.data.frame(quality), row.names = FALSE)
cat("\n")
write.csv(quality, file.path(out_dir, "front_quality.csv"), row.names = FALSE)

# ── 7. What is actually on the Pareto fronts? ────────────────────────────────
have_params <- intersect(PARAMS, names(trials))
if (length(have_params)) {
  pareto_cfgs <- trials |>
    filter(on_pareto) |>
    select(pid, phaseStep, Phase, all_of(have_params), all_of(OBJECTIVES), hv_r)
  cat("Pareto-optimal configurations (", nrow(pareto_cfgs), " rows)\n", sep = "")
  print(as.data.frame(pareto_cfgs), row.names = FALSE)
  write.csv(pareto_cfgs, file.path(out_dir, "pareto_configs.csv"), row.names = FALSE)

  if ("pattern" %in% have_params) {
    cat("\npattern share on the Pareto fronts vs overall:\n")
    print(trials |> group_by(pattern) |>
            summarise(overall = n(), on_pareto = sum(on_pareto), .groups = "drop") |>
            mutate(pareto_rate = round(on_pareto / overall, 3)) |> as.data.frame(),
          row.names = FALSE)
    cat("\nNOTE: for pattern == \"constant\", space.canonicalize() pins",
        paste(DEAD_FOR_CONSTANT, collapse = " and "), "to sentinels\n",
        "(19.95 s and 1 Hz). Those are not measurements — exclude them from any\n",
        "parameter-effect analysis restricted to constant rows.\n")
  }
  cat("\n")
}

# ── 8. Figures ───────────────────────────────────────────────────────────────
save_fig <- function(p, name, columns = 1) {
  ok <- try(colleyRstats::save_paper_figure(p, filename = file.path(out_dir, name),
                                            columns = columns), silent = TRUE)
  if (inherits(ok, "try-error")) {
    ggplot2::ggsave(file.path(out_dir, name), p, width = 7, height = 5, dpi = 300)
  }
  cat("  wrote", file.path(out_dir, name), "\n")
}

cat("figures\n")

# 8a. Objective scores over trials, with the Sampling/Optimization annotation.
for (obj in OBJECTIVES) {
  p <- try(colleyRstats::plot_mobo2(
    data = as.data.frame(trials), x = "phaseStep", y = obj,
    phaseCol = "Phase",
    fillColourGroup = if ("pattern" %in% names(trials)) "pattern" else "",
    ytext = obj, horizontalLinePosY = 0.95), silent = TRUE)
  if (!inherits(p, "try-error")) save_fig(p, paste0("mobo_", obj, ".pdf"))
  else cat("  (plot_mobo2 failed for", obj, "- needs both phases present)\n")
}

# 8b. Hypervolume convergence, one line per participant.
p_hv <- ggplot(trials, aes(x = phaseStep, y = hv_r, group = pid, colour = pid)) +
  geom_vline(xintercept = N_SOBOL + 0.5, linetype = "dashed", linewidth = 0.4) +
  geom_step(linewidth = 0.7) +
  geom_point(size = 1.2) +
  annotate("text", x = N_SOBOL / 2, y = -Inf, label = "Sampling",
           vjust = -0.8, fontface = "bold", size = 4) +
  annotate("text", x = N_SOBOL + (max(trials$phaseStep) - N_SOBOL) / 2, y = -Inf,
           label = "Optimization", vjust = -0.8, fontface = "bold", size = 4) +
  labs(x = "Trial", y = "Hypervolume", colour = "Participant") +
  colley_theme()
save_fig(p_hv, "hypervolume_convergence.pdf")

# 8c. Objective space, Pareto-optimal points highlighted.
p_front <- ggplot(trials, aes(x = .data[[OBJECTIVES[1]]], y = .data[[OBJECTIVES[2]]])) +
  geom_point(aes(shape = Phase), colour = "grey55", alpha = 0.7, size = 2) +
  geom_point(data = filter(trials, on_pareto), aes(colour = pid), size = 3) +
  geom_step(data = filter(trials, on_pareto) |> arrange(pid, .data[[OBJECTIVES[1]]]),
            aes(colour = pid, group = pid), direction = "vh", linewidth = 0.5) +
  labs(x = OBJECTIVES[1], y = OBJECTIVES[2], colour = "Participant",
       shape = "Phase") +
  colley_theme()
save_fig(p_front, "pareto_front.pdf")

# ── 9. Machine-readable trial table ──────────────────────────────────────────
write.csv(trials |> select(-all_of(neg_names)),
          file.path(out_dir, "trials_annotated.csv"), row.names = FALSE)

cat("\n=============================================================\n")
cat("done — outputs in", normalizePath(out_dir, winslash = "/"), "\n")
cat("=============================================================\n")
