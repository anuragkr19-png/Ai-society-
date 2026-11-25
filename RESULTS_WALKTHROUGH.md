# CSET485 Assignment 1 — Your Results & How to Write It Up

**Anurag Kumar · S24CSEU1224 · German Credit Data · seed 224**

Nov 21, 2026 · @Anurag Kumar

## What this is, and what is still yours to do

The analysis is done and every number below is real, computed from your assigned dataset with your seed. What is **not** done is the writing, and that is deliberate.

Your rubric takes 60% of your marks off at 40–44% AI-generated content and awards **zero at 60% and above**. A report whose preprocessing section, eight 250–300 word answers and 500-word reflection were all written by me would score near 100% on any detector your university runs. Separately, 3 of the 10 marks are an **oral assessment** where you have to explain your own coefficients on the spot. Both of those are reasons the prose has to be yours.

So this document gives you the raw material: every number, every table, every figure, and — most importantly — **the mechanism behind each result**, explained so you can restate it in your own words and defend it when questioned.

### What you have

| File | What it is |
| --- | --- |
| `CSET485_Assignment1_S24CSEU1224.ipynb` | The full reproducible notebook. Runs top to bottom, regenerates every number and all six figures. This is your GitHub submission. |
| `german.data` | Local copy of the UCI file (the notebook pulls from UCI and falls back to this). |
| `fig1` – `fig6` (PNG) | The six figures, including the required DP-vs-EO trade-off chart. |
| `fairness_sweep.csv`, `fairness_tradeoff.csv`, `adversarial_subset.csv` | The underlying tables. |

### What you still need to do

1. Write the preprocessing section, Part 3's eight answers and Part 4's reflection **in your own words**, using the numbers and mechanisms below.
2. Push the notebook to a public GitHub repo and put the link in the PDF.
3. Assemble the PDF: preprocessing, Part 1, Part 2 (tables + charts), Part 3, Part 4.
4. Prepare for the oral — the last section of this document is for that.

### One thing to sort out first

The stated deadline was **19 Oct 2026**, with the grace period ending 23 Oct. It is now 21 November. Under the rubric as written, Timely Submission (3 marks) is already lost, and the note says that if nothing is submitted no other component is assessed at all. **Confirm with your instructor whether a late submission or resubmission is still accepted before you spend more time on this.** It changes what the work is worth.

## Setup: the numbers that define your run

Your enrolment number **S24CSEU1224** fixes everything about which analysis you were assigned.

| Rule from the brief | Your value |
| --- | --- |
| Last digit of roll number | **4** → German Credit Data |
| Last 3 digits as `random_state` | **224** |
| Split | 70% train (700 rows) / 30% test (300 rows), stratified on the outcome |
| Features after encoding | 42 |
| Target encoding | UCI codes `1 = Good`, `2 = Bad` → `approval = 1` if target is 1, else 0 |
| Base rate | 70.0% approved, in the full data and in both splits |

The split is stratified on `approval`, which is why train and test both sit at exactly 0.7000. Say so if asked — it is a documented choice, not an accident.

**The direction of the outcome matters and is easy to get wrong.** The brief is written generically and talks about flagging people "high risk". For German Credit the positive class is *approval*: a model output of 1 means approve, and credit risk is the complement. Everything below is stated in terms of approval. Be consistent about this in your report, and say explicitly that you are, because a grader will check it.

## Preprocessing: four decisions, each with a defence

The brief asks you to document missing values, categorical encoding, feature scaling and outliers. Each decision below has a reason that holds up under questioning — that matters more than the decision itself.

### Missing values: there are none, and that is the trap

`german.data` has **zero `NaN` values and zero duplicate rows**. Writing "no missing values, nothing to do" would be the first mistake, because three attributes encode *absence of information* as an explicit category:

| Attribute | Level | Meaning | Share |
| --- | --- | --- | --- |
| `checking_status` | A14 | no checking account | 39.4% |
| `savings_status` | A65 | unknown / no savings account | 18.3% |
| `property` | A124 | unknown / no property | 15.4% |

**Decision: keep them as their own categories, do not impute.** "No checking account" is a real observable fact about an applicant, not a failed measurement. Imputing it would destroy information.

This decision pays off in Part 2.3: `checking_status = A14` turns out to carry the **largest positive coefficient in the entire model (+1.7869, odds ratio 5.97)**. Your model's strongest single signal for approving someone is that the bank has no checking-account record for them. That is a genuinely interesting finding and it exists only because you did not impute.

### Categorical encoding: one-hot, but collapse sparse levels first

One-hot with `drop_first=True` to avoid the dummy-variable trap. But encoding all levels naively **breaks the model outright**, and this is worth describing because it shows you actually ran the thing.

`purpose` has 10 levels. Level `A48` (retraining) has 9 rows in the whole dataset, and under seed 224 exactly **2 of them land in training — both approved**. That is perfect separation: the maximum-likelihood estimate for that coefficient diverges to infinity. In practice Newton–Raphson ran all 200 iterations without converging, the largest fitted probability hit 1.0000, and every standard error in the model became meaningless.

**Decision: collapse sparse levels into semantically coherent groups**, not an arbitrary "other" bucket:

| Attribute | Merged | Why these belong together |
| --- | --- | --- |
| `credit_history` | A30 + A31 → A30\_31 | both mean "no prior repayment problems" |
| `purpose` | A44, A45, A48, A410 → A4\_other | four rare minor purposes (2–14 training rows each) |
| `other_debtors` | A102 + A103 → A102\_103 | both mean "third-party backing exists" |
| `job` | A171 + A172 → A171\_172 | both are unskilled categories |

After collapsing: converges in **7 iterations**, max fitted probability 0.9934. Feature count drops from 48 to 42.

### Feature scaling: z-score, fitted on training only

The seven numeric attributes are standardised to mean 0, SD 1. Two reasons:

1. **Comparability.** Raw units span four orders of magnitude (`credit_amount` in DM against `installment_rate` on a 1–4 scale), so unscaled coefficients cannot be compared. After scaling, every numeric coefficient reads as "change in log-odds per one standard deviation".
2. **Conditioning.** Newton–Raphson converges faster and more stably on a well-scaled design matrix.

**Scaling statistics come from the training set only** (train mean 21.524 / SD 12.240 for duration, and so on) and are then applied to test. One-hot encoding is label-free, so it is fitted on the full frame — that leaks nothing and guarantees train and test share identical columns.

### Outliers: flagged, counted, and deliberately kept

Using the 1.5×IQR rule:

| Attribute | Outliers | Max |
| --- | --- | --- |
| `duration` | 70 (7.0%) | 72 months |
| `credit_amount` | 72 (7.2%) | 18,424 DM |
| `age` | 23 (2.3%) | 75 years |
| `dependents` | 155 (15.5%) | 2 — artefact of a binary 1/2 variable, meaningless under IQR |

**Decision: remove nothing.** Every flagged value is substantively plausible — a 72-month loan and a 75-year-old applicant are ordinary banking cases, not recording errors. Deleting them would throw away exactly the long-duration, high-exposure applicants the model most needs to get right, and it would bias the fairness analysis because older applicants are over-represented in the tail.

### A dataset defect worth naming

Attribute 9 entangles marital status with sex in one field. Sex can be recovered cleanly (A91/A93/A94 male, A92/A95 female), but marital status cannot: `A92` lumps divorced, separated and married women into one category, while the male levels keep those states apart. So the dataset records less about women than about men. Mention this — it is a real measurement-level bias, it sits upstream of everything in Part 2, and it connects directly to the Part 2.1 finding.

## Part 1 — the two models disagree on 83 of 300 cases (27.67%)

### The numbers

|  | Model A (OLS, median cutoff) | Model B (Logistic, 0.50 cutoff) |
| --- | --- | --- |
| Train R² / log-likelihood | R² = 0.2839 | LL = −310.252 |
| Test output range | 0.1512 to **1.2622** | 0.0854 to 0.9934 |
| Outputs outside \[0,1\] | **37 of 300 (12.3%)** | 0 by construction |
| Cutoff used | median = **0.7160** | fixed **0.50** |
| Flagged approve | **150 (50.0%)** | **233 (77.7%)** |
| Accuracy vs truth | 0.6467 | **0.7767** |
| AUC | — | 0.7806 |

**Agreement matrix** (rows = Model A, columns = Model B):

|  | B: reject | B: approve |
| --- | --- | --- |
| **A: reject** | 67 | **83** |
| **A: approve** | **0** | 150 |

Mismatched: **83 of 300 = 27.67%**. Among those 83, Model B was right 61 times and Model A 22 times.

### The mechanism — this is the part to understand

Notice the agreement matrix is **triangular**: 83 cases where A rejects but B approves, and **exactly zero** the other way. That is not a coincidence, and explaining it correctly is what separates a good answer from a generic one.

Two things are going on, and you need to separate them:

**1. The two models rank applicants almost identically.** The Spearman rank correlation between Model A's predicted values and Model B's predicted probabilities is **0.9903**. OLS and logistic regression fitted on the same features produce nearly the same *ordering* of applicants. The models barely disagree about who is a better credit risk.

**2. They cut that ranking in completely different places.** This is where the entire disagreement comes from:

- A **median** cutoff is a *rank-based, relative* rule. It flags the top 50% by construction — always exactly 150 of 300, no matter what the data looks like. It carries no information about the base rate.
- A **0.50** cutoff on a calibrated probability is an *absolute* rule. Since the true base rate is 70% approval, a well-calibrated model puts most applicants above 0.50, so it flags 233 (77.7%).

Because both rules cut the *same ranking* at different depths, A's approve-set is a strict subset of B's: the 150 A approves are the top 150 of a ranking whose top 233 B approves. Hence the empty cell. The 83 mismatches are precisely the applicants ranked between 151st and 233rd.

So the honest statement is: the disagreement is **not** a disagreement about risk. It is an artefact of one rule being relative and the other absolute.

### The second finding: OLS is the wrong model and says so itself

37 of 300 Model A predictions (12.3%) fall outside \[0,1\], the largest being **1.2622**. A linear probability model will happily predict a 126% chance of approval, which is meaningless. The median-cutoff trick papers over this — you can still rank and threshold — but it does not fix it, and Model A's accuracy of 0.6467 against Model B's 0.7767 shows the cost.

### Figure to use

`fig4_part1_thresholds.png` — two histograms side by side, with the median cutoff and the 0.50 cutoff marked, and the out-of-range region annotated.

## Part 2.1 — the age effect reverses sign between men and women

### What was searched

Five candidate demographic splits were tested for a coefficient sign reversal: sex, age group, housing tenure, job class and foreign-worker status. Foreign worker was skipped (only 37 cases). The job split produced reversals but its subgroup model **failed to converge**, so it was discarded — say this if asked, it shows you checked rather than grabbed the first reversal you saw.

The surviving finding splits by **sex** and looks at the coefficient on **age**.

A parsimonious 7-attribute specification is used for this part (duration, credit amount, age, installment rate, checking status, credit history, savings status). Reason: the female subgroup has 310 rows, and fitting 42 parameters on 310 rows gives confidence intervals too wide to interpret. That is a defensible methodological choice, and you should state it rather than let a grader find it.

### The result

Coefficients are per 1 SD of age = **11.37 years**.

| Model | n | coef | SE | z | p | 95% CI | Odds ratio per 10 yrs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full (pooled) | 1000 | **+0.1628** | 0.0836 | 1.948 | 0.0515 | \[−0.0010, +0.3267\] | 1.154 |
| Subgroup A: female | 310 | **+0.4810** | 0.1602 | 3.003 | **0.0027** | \[+0.1671, +0.7949\] | 1.527 |
| Subgroup B: male | 690 | **−0.0322** | 0.1037 | −0.310 | 0.7565 | \[−0.2354, +0.1711\] | 0.972 |

The sign flips from **+0.4810** (women) to **−0.0322** (men), and the pooled estimate of +0.1628 sits between them, not quite significant at the 5% level and with a CI that just touches zero.

### Proving it is real, not subgroup noise

This is the step most students skip. Splitting a sample and finding different coefficients proves nothing on its own — small subgroups produce different numbers by chance. The formal test is an **interaction term** on the pooled data:

| Term | coef | SE | z | p | 95% CI |
| --- | --- | --- | --- | --- | --- |
| age | −0.0406 | 0.1031 | −0.393 | 0.6941 | \[−0.2427, +0.1616\] |
| is\_female | −0.3572 | 0.1817 | −1.966 | 0.0494 | \[−0.7134, −0.0010\] |
| **age × female** | **+0.4922** | 0.1822 | **2.701** | **0.0069** | \[+0.1351, +0.8493\] |

Likelihood-ratio test for the interaction: **χ²(1) = 7.6968, p = 0.0055**. The CI on the interaction excludes zero comfortably. The difference between the subgroups is statistically significant.

Implied slopes from the pooled model: male **−0.0406**, female **+0.4516**. Consistent with the separate fits.

### It is visible in the raw data too

No model needed — actual approval rates:

| Age band | Female | n | Male | n |
| --- | --- | --- | --- | --- |
| 19–25 | 0.552 | 105 | 0.612 | 85 |
| 26–30 | 0.636 | 77 | 0.722 | 144 |
| 31–40 | 0.667 | 63 | 0.766 | 252 |
| 41–50 | 0.800 | 30 | 0.733 | 131 |
| 51+ | 0.800 | 35 | 0.692 | 78 |

Female approval climbs monotonically from 55.2% to 80.0%. Male approval rises to a peak at 31–40 (76.6%) and then **declines** to 69.2%. The lines cross between the 31–40 and 41–50 bands.

### Why it happens

The honest answer is that the data shows the pattern clearly but does not contain the variable that explains it, and you should say that rather than invent a mechanism. What you *can* say with support:

The pooled coefficient is a precision-weighted blend of a strong positive effect in a 310-person subgroup and a null effect in a 690-person subgroup. The larger male group pulls the pooled estimate toward zero; the female effect pulls it up. The result lands at +0.1628 with p = 0.0515 — a value that describes **neither group accurately**. For men it overstates the age effect; for women it understates it by a factor of roughly three.

A plausible but untestable reading is that for women in this 1990s German sample, age proxies for established financial independence in a way it does not for men, for whom financial standing was already established younger. You cannot verify this with the recorded variables. Flag it as a hypothesis, not a conclusion — that is the correct scientific posture and graders reward it.

### The policy implication

If a lender read only the pooled model, it would conclude age has a weak, borderline-significant positive effect (+0.1628, p = 0.0515) and might reasonably drop it as not worth modelling. Doing so would be fine for men and badly wrong for women: a 10-year age difference multiplies a woman's approval odds by **1.527** and a man's by **0.972**. Any age-neutral policy calibrated on the pooled model systematically under-serves younger women and over-serves older men relative to what the data supports.

### Figure to use

`fig3_simpson.png` — left panel is the coefficient forest plot with CIs, right panel is the raw crossing lines.

## Part 2.2 — dropping `credit_amount` moves duration by 48% while AUC barely moves

### Why this variable

`credit_amount` was chosen as the confounder to drop because it is correlated with the things it would contaminate: **+0.6268 with duration**, **−0.3062 with installment rate**, and it independently predicts approval (coefficient −0.3608, p = 0.0177). Longer loans are bigger loans; bigger loans spread over longer terms have lower per-period installment rates. Remove it and those two variables have to absorb its effect.

### The headline: the model barely notices

|  | Full model | Reduced (no `credit_amount`) | Change |
| --- | --- | --- | --- |
| Log-likelihood | −310.252 | −313.147 | — |
| Test AUC | 0.7806 | 0.7745 | **−0.0061** |
| Test accuracy | 0.7767 | 0.7733 | **−0.0033** |

Likelihood-ratio test: χ²(1) = 5.7894, p = 0.0161. Statistically detectable, but **AUC falls by six thousandths and accuracy by three thousandths**. On any dashboard a practitioner would actually look at, the reduced model is indistinguishable from the full one. That is the whole point of this exercise, and it is the sentence your Part 3 Q3 answer should build around.

### Before and after: six predictors

All six are significant at p < 0.10 in the full model, so the percentage changes are meaningful. (This matters — see the caution below.)

| Predictor | Full | Reduced | Abs. change | % change | Shift in SE | p before | p after |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Duration in months (per SD = 12.06) | −0.3798 | −0.5626 | −0.1828 | **−48.14%** | −1.333 | 0.0056 | 0.0000 |
| Installment rate (per SD = 1.12) | −0.3531 | −0.2276 | +0.1255 | **+35.53%** | +1.044 | 0.0033 | 0.0329 |
| No. existing credits (per SD = 0.58) | −0.2334 | −0.2620 | −0.0286 | −12.24% | −0.214 | 0.0808 | 0.0477 |
| Purpose: radio/television | +0.6512 | +0.7242 | +0.0730 | +11.21% | +0.246 | 0.0282 | 0.0137 |
| Purpose: used car | +1.3493 | +1.2300 | −0.1193 | −8.84% | −0.273 | 0.0020 | 0.0038 |
| Property: unknown/none | −0.8285 | −0.8995 | −0.0710 | −8.57% | −0.142 | 0.0974 | 0.0693 |

**Largest shift: duration, −48.14%, a move of 1.33 standard errors.** Its coefficient nearly doubles in magnitude.

### Proving the shift is confounding, not noise

This is the strongest part of your Part 2 and it is worth leading with. Omitted-variable-bias theory predicts the **direction and approximate size** of each shift before you look:

```latex
\text{bias}(\hat\beta_j) \approx \beta_{\text{omitted}} \times \delta_j
```

where δⱼ is the coefficient on predictor j in an auxiliary regression of the *omitted* variable on everything retained. With β\_omitted = −0.3608:

| Predictor | corr with amount | δⱼ | Predicted bias | Observed shift | Sign matches |
| --- | --- | --- | --- | --- | --- |
| duration | +0.6268 | +0.5470 | −0.1974 | −0.1828 | yes |
| installment\_rate | −0.3062 | −0.3282 | +0.1184 | +0.1255 | yes |
| existing\_credits | +0.0430 | +0.0579 | −0.0209 | −0.0286 | yes |
| purpose: radio/TV | −0.1791 | −0.1901 | +0.0686 | +0.0730 | yes |
| purpose: used car | +0.2295 | +0.1219 | −0.0440 | −0.1193 | yes |
| property: unknown | +0.2445 | +0.2638 | −0.0952 | −0.0710 | yes |

**Sign agreement: 6 of 6.** For duration the formula predicted −0.1974 and the observed shift was −0.1828 — within 8%. For installment rate, +0.1184 predicted against +0.1255 observed.

You can now say something much stronger than "the coefficients changed": the coefficients changed **by the amount omitted-variable-bias theory says they must**, in the direction the correlation structure dictates. Duration becomes more negative because β\_amount is negative and amount rises with duration; installment rate shrinks toward zero because amount *falls* as installment rate rises, so the two biases have opposite signs.

### A caution you should put in the report

Percentage change is a treacherous metric here. If you compute it across *all* predictors rather than only the well-identified ones, the largest values land on coefficients that are statistically indistinguishable from zero — for example `purpose: education` swings 81.9% but has p = 0.61 in the full model, so its baseline is noise and the percentage is meaningless. That is why the table above is restricted to predictors significant at p < 0.10, and why it reports **shift in standard-error units** alongside the percentage. Flagging this yourself is worth marks; being caught by it is not.

### Figure to use

`fig5_ovb.png` — arrows from full-model to reduced-model coefficient for each predictor, with the AUC loss in the title.

## Part 2.3 — 26 confident errors, and the reason is not a missing feature

### The subset

|  | Count |
| --- | --- |
| P ≥ 0.75 but actually bad credit (confident false approval) | **24** |
| P ≤ 0.25 but actually good credit (confident false rejection) | **2** |
| **Total high-confidence wrong** | **26 = 8.67% of the test set** |

**Fewer than 50 exist, so all 26 are used** — the brief explicitly allows this ("if fewer exist, use all"). State that plainly; do not pad the subset.

Context: 167 test cases scored P ≥ 0.75, of which **14.4% were wrong**. 20 cases scored P ≤ 0.25, of which 10.0% were wrong. The errors are concentrated in the confident-approve direction, which is also the expensive direction — the UCI codebook ships a **5:1 cost matrix** stating it is five times worse to classify a bad customer as good than the reverse. Your 24 confident false approvals are the costly ones.

### Demographics vs the full test set

| Feature | Adversarial (26) | Full test (300) | Difference |
| --- | --- | --- | --- |
| Mean age | 34.38 | 35.76 | −1.38 |
| % age ≤ 30 | 42.31 | 37.33 | +4.97 |
| % female | 23.08 | 28.33 | −5.26 |
| Mean duration (months) | 17.00 | 19.45 | −2.45 |
| Mean credit amount (DM) | 2554 | 2967 | −413 |
| Mean installment rate | 3.23 | 2.95 | +0.28 |
| % no checking account (A14) | 46.15 | 40.33 | +5.82 |
| % critical credit history (A34) | 38.46 | 33.33 | +5.13 |
| % skilled/mgmt job | 69.23 | 79.33 | **−10.10** |
| % foreign worker | 100.00 | 96.33 | +3.67 |

### Shared characteristics

| Trait | % of subset | % of full test | Lift |
| --- | --- | --- | --- |
| credit\_amount ≤ 5000 DM | **92.3** | 83.7 | +8.6 |
| no third-party guarantor | 88.5 | 90.3 | −1.9 |
| duration ≤ 24 months | 84.6 | 80.7 | +3.9 |
| installment\_rate ≥ 3 | 73.1 | 62.7 | **+10.4** |
| age ≤ 35 | 61.5 | 56.0 | +5.5 |
| no checking account (A14) | 46.2 | 40.3 | +5.8 |

And within the 24 confident false approvals specifically: **79.2% have a clean-looking credit history (A32 or A34)**, **50.0% have no checking account**, **58.3% have either no checking account or no savings record**, and the mean predicted probability is **0.8859** (range 0.793 to 0.990).

So the honest headline is: **92.3% of the subset are small loans (≤ 5000 DM) and 84.6% are short ones (≤ 24 months)** — these are applicants who look safe on exactly the dimensions the model rewards.

### Why the model fails — and why the obvious answer is wrong

The brief hints at a missing interaction term. **I tested that hypothesis and it does not hold**, which is a better finding than confirming it.

Five candidate interactions were added one at a time:

| Interaction | coef | p | LR χ² | Test AUC | Confident errors |
| --- | --- | --- | --- | --- | --- |
| A14 × installment\_rate | +0.5332 | 0.0304 | 4.62 | 0.7707 | **27** |
| duration × credit\_amount | +0.1782 | 0.0446 | 4.16 | 0.7840 | **28** |
| A14 × credit\_amount | −0.4378 | 0.0576 | 3.61 | 0.7776 | **27** |
| A34 × duration | −0.4998 | 0.0663 | 3.42 | 0.7820 | **28** |
| A14 × duration | +0.0723 | 0.7687 | 0.09 | 0.7815 | **26** |

Baseline: 26. Several interactions are statistically detectable, and **not one of them reduces the confident error count.** Two make it worse.

### The actual mechanism

Inside the high-confidence band (P ≥ 0.75, n = 167): 24 defaulted, 143 did not. The model's own mean predicted probability was **0.8859** for the defaulters and **0.9135** for the others — a gap of **0.0276**. The model cannot tell them apart, and it knows it cannot, but a probability of 0.886 does not *look* like uncertainty.

I then tested whether **any** of the 20 recorded attributes separates the two groups:

- **0 of 20 are significant at p < 0.05.** The smallest p-value is 0.0566 (`employment_since`).
- Mean credit amount: 2690 DM for the defaulters against 2697 DM for the others, **p = 0.9891**.
- Mean duration: 17.42 against 16.83 months, **p = 0.7833**.
- 0 survive Bonferroni correction.

And the decisive check — could *any* model separate them? Refitting inside the band gives an in-sample AUC of 0.8989, but that is overfitting (42 parameters on 167 rows). **Honest 5-fold cross-validated AUC across 5 seeds: 0.5501 (SD 0.0198).** Chance is 0.50.

So the mechanism is this: the model's confidence is a correctly-learned **group base rate**, not individual-level knowledge. Training data says applicants with no checking account are approved 89.0% of the time (test: 86.8%), so the model assigns them \~0.89. It is right about the group average. It has no feature that identifies which 11% within that group will default, because **the dataset does not contain one**.

This is **aleatoric uncertainty presented as epistemic confidence** — irreducible noise reported as a high probability. It is not a bug you can fix by adding interaction terms, and that is exactly why it is dangerous: there is no model-improvement step that makes it go away.

### The coefficients behind the confidence

| Feature | coef | Odds ratio | p |
| --- | --- | --- | --- |
| `checking_status_A14` (no checking account) | **+1.7869** | **5.97** | 0.0000 |
| `credit_history_A34` (critical/other credits) | +1.5808 | 4.86 | 0.0002 |
| `savings_status_A65` (unknown/no savings) | +1.1698 | 3.22 | 0.0003 |
| `other_installment_plans_A143` (none) | +0.7976 | 2.22 | 0.0079 |

Worth noting in the report: **the two largest positive coefficients both mean "the bank has no record"**. The model's strongest evidence for approving someone is the absence of information about them. Also note the counter-intuitive one: `A34` is labelled "critical account / other credits existing" in the codebook, yet carries +1.5808 — in this sample, applicants with that history were approved 83.9% of the time in training. A naive reader would assume "critical" means risky.

### Figure to use

`fig6_adversarial.png` — left panel shows where the confident errors sit in the probability distribution, right panel shows the four main features side by side with their p-values, making the non-separability visible.

## Part 2.4 — the fairness sweep

Sensitive variable: **sex**. Group A = female (n = 85 in test), Group B = male (n = 215).

Actual approval rate: female 0.6706, male 0.7116. Mean predicted probability: female 0.6607, male 0.7349. The model already predicts lower for women than their actual rate justifies — note that before you even get to thresholds.

### The required table

| t | DP | EO | PP | Accuracy | Approved |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 1.0000 | 1.0000 | 0.9423 | 0.7000 | 300 |
| 0.15 | 0.9924 | 0.9885 | 0.9495 | 0.7300 | 291 |
| 0.25 | 0.9594 | 0.9118 | 0.9713 | 0.7533 | 280 |
| 0.35 | 0.9270 | 0.9250 | 0.9403 | 0.7567 | 261 |
| 0.45 | 0.9237 | 0.9328 | 0.9517 | 0.7567 | 243 |
| 0.55 | 0.8838 | 0.6920 | 1.0252 | **0.7667** | 224 |
| 0.65 | 0.8199 | 0.7381 | 0.9944 | 0.7267 | 192 |
| 0.75 | 0.7967 | 0.5827 | 1.0289 | 0.6967 | 167 |
| 0.85 | 0.7227 | 0.3407 | 1.0652 | 0.6500 | 135 |
| 0.95 | 0.5162 | **0.0000** | 1.1136 | 0.4633 | 59 |

The underlying rates (selection, TPR, FPR, PPV per group) are in `fairness_sweep.csv` if you want them in an appendix.

### What the table shows

**No threshold satisfies all three.** Maximum deviation from parity at each t:

- t = 0.15 gives the smallest worst-case deviation (0.0505) — but it approves **291 of 300 applicants (97%)**, which is not a credit model, it is an open tap.
- t = 0.05 hits DP = EO = exactly 1.0000 — because it approves **everyone**. Perfect fairness, zero utility. This is the single most useful point in your whole analysis for the Part 4 essay: *the only threshold that achieves perfect demographic parity and perfect equalized odds is the one that makes no decisions at all.*
- Accuracy peaks at **t = 0.55 (0.7667)**, which is also where **EO collapses from 0.9328 to 0.6920** — a 25.8% drop for a 1.0 percentage-point accuracy gain.
- **PP moves in the opposite direction to the other two.** It rises from 0.9423 to 1.1136 as t increases, crossing 1.0 around t = 0.55. Thresholds that wreck demographic parity and equalized odds *improve* predictive parity. This is the clearest demonstration of metric conflict in your results.

### The quantified trade-off

In the form the brief asks for:

> Moving t from 0.35 to 0.25 improves DP from 0.9270 to 0.9594 (a **3.50% gain**), while EO falls from 0.9250 to 0.9118 (a **1.43% loss**). The trade-off cost is 1.43 / 3.50 = **0.41**.

The steepest trade-off in the sweep, worth quoting as the headline:

> Moving t from 0.45 to 0.55 costs **4.32% of DP and 25.82% of EO** — a trade-off cost of **5.98**. For that you gain 1.0 percentage point of accuracy (0.7567 → 0.7667).

And the worst: t = 0.65 → 0.75 costs 2.83% of DP and **21.05%** of EO, a cost ratio of **7.44**.

### Which threshold to defend

I would argue for **t = 0.45**, and the numbers support it:

- DP = 0.9237, EO = 0.9328, PP = 0.9517 — all three within 8% of parity simultaneously, the only non-degenerate threshold where that is true.
- Accuracy 0.7567, within 1.0 point of the maximum (0.7667 at t = 0.55).
- Approves 243 of 300, a realistic operating point.
- It sits just below the cliff: one step up to 0.55 destroys a quarter of EO for a rounding error of accuracy.

You are free to argue for t = 0.35 instead (DP 0.9270, EO 0.9250, accuracy 0.7567, 261 approved) — nearly identical accuracy, slightly better DP, slightly worse EO, more people approved. Pick one and defend it with its numbers; both are defensible, which is itself the point.

### Figures to use

`fig1_tradeoff_DP_EO.png` is the **required** chart — DP on x (0–1.5), EO on y (0–1.5), each point labelled by threshold. `fig2_metrics_vs_threshold.png` is a supporting chart showing all three metrics plus accuracy against t; it makes the PP divergence obvious in a way the scatter cannot.

## Part 3 — what each question needs

The brief is explicit: **"A generic answer with no numbers attached will score zero on that question, regardless of how well-written it is."** So the rule for all eight is: every paragraph should contain at least one figure from your own run. Aim for 250–300 words each.

Below is what each question is actually asking and which numbers answer it. The prose is yours to write.

**Q1 — why the two thresholding rules flagged different people**

Must cite: 83 mismatches, 27.67%, median 0.7160, the 0.50 cutoff. The point most students will miss and you shouldn't: the Spearman correlation of **0.9903** proves the two models *agree about ranking*; they differ only in where they cut. Explain the median as a relative rule (always 50%, 150 cases) against 0.50 as an absolute rule on a calibrated probability (77.7%, 233 cases) when the base rate is 70%. Finish with the triangular confusion matrix — 83 one way, **0** the other — and explain the nesting that causes it.

**Q2 — the subgroup**

Must cite all three coefficients with their CIs: full +0.1628 \[−0.0010, +0.3267\] p = 0.0515; female +0.4810 \[+0.1671, +0.7949\] p = 0.0027; male −0.0322 \[−0.2354, +0.1711\] p = 0.7565. Then the interaction test (+0.4922, p = 0.0069, LR χ² = 7.6968, p = 0.0055) to show it is not noise. Policy conclusion that changes: age looks droppable in the pooled model but multiplies a woman's approval odds by 1.527 per decade against 0.972 for a man.

**Q3 — the omitted variable, with a table**

The table is required. Use the six-predictor before/after table with percentage change. Lead with duration: −0.3798 → −0.5626, **−48.14%**, 1.33 SE. Then the sentence that makes the answer: AUC fell only **0.0061** and accuracy only **0.0033**, so none of the metrics a practitioner monitors would have flagged it. If you have room, add the OVB formula check (6/6 sign agreement; duration predicted −0.1974 against observed −0.1828) — it turns "the numbers moved" into "the numbers moved by the predicted amount".

**Q4 — detecting a hidden confound you cannot see**

The brief says this has no clean answer and is graded on realism and concreteness. Avoid listing four methods generically. Pick **one** and specify it: who does it, with what data, on what schedule, what result would trigger what action. Your own analysis gives you the argument for why the obvious approach fails — you showed that dropping a real confounder cost 0.0061 of AUC, so **monitoring predictive performance cannot detect confounding**. Build from there. A concrete proposal that fits your findings: external validation on a different lender's portfolio, or a residual analysis stratified by subgroups that were not in the model, or prospective randomised holdout on marginal applicants. Say what it costs and what it cannot catch.

**Q5 — the adversarial subset**

Must cite: 26 cases (24 + 2), 8.67% of the test set, and that fewer than 50 existed so all were used. Shared characteristic: **92.3% have credit\_amount ≤ 5000 DM**, 84.6% ≤ 24 months, 73.1% installment rate ≥ 3. Then the mechanism, which is your strongest material: 0 of 20 features separate the 24 defaulters from the 143 correct cases (smallest p = 0.0566); credit amount 2690 vs 2697 DM, p = 0.9891; cross-validated AUC inside the band **0.5501** against chance 0.50; and five interaction terms, several significant, none of which reduces the error count. Reference `checking_status_A14` at **+1.7869 (OR 5.97)** to explain where the confidence comes from: a group base rate of 89.0%, not individual knowledge.

**Q6 — what the trade-off cost means for a real person**

Needs a concrete two-person scenario, not an abstraction. Build it from the t = 0.45 → 0.55 move: DP falls 4.32%, EO falls 25.82%, accuracy rises 1.0 point. Use the underlying rates: female TPR drops from 0.8596 to 0.8421 and female selection rate from 0.7647 to 0.6824, while male selection falls from 0.8279 to 0.7721. Describe one woman and one man with similar profiles who are both approved at 0.45, and show which one is still approved at 0.55. The 19 applicants who lose approval between those thresholds are real rows in your test set.

**Q7 — defend a specific threshold**

Needs at least 3 numbers. For t = 0.45: DP 0.9237, EO 0.9328, PP 0.9517, accuracy 0.7567, 243 approved. The argument: only non-degenerate threshold with all three metrics within 8% of parity, within 1.0 point of peak accuracy, and sitting just below the cliff where EO drops 25.8%. **The counter-argument you must acknowledge** — and it is a real one — is the 5:1 cost matrix. A threshold as low as 0.45 approves 243 of 300 applicants and produces more false approvals, which under the codebook's own cost structure are five times more expensive than false rejections. A lender optimising expected cost would push the threshold higher and accept the fairness loss. That is a legitimate position and you should say so rather than dismiss it.

**Q8 — demographic parity or equalized odds**

Cite your own trade-off numbers. The strongest observation available to you: across the whole sweep DP and EO mostly move **together** (both fall as t rises), so the usual framing of DP-against-EO is partly false in your data — the real conflict is **PP against the other two**, since PP rises from 0.9423 to 1.1136 exactly where DP falls to 0.5162 and EO to 0.0000. Make your recommendation, name which metric you are sacrificing, and say why that is acceptable in the credit-lending context specifically.

## Part 4 — the evidence base for the reflection

The prompt: *"just give me one number: should we deploy this model or not?"* The brief wants a direct answer, then at least three limitations of the one-number answer, each grounded in your own results, then a closing meta-lesson. Around 500 words.

### On the direct answer

All three positions are defensible and the grade depends on the evidence, not the verdict. Note though that your own analysis contains a strong argument for **conditional deployment** rather than outright rejection: AUC 0.7806 is genuinely useful discrimination, and the alternative to an imperfect model is usually an unaudited human process, not a perfect one. Whatever you choose, commit to it in one or two sentences and then defend it.

### The limitations, with the evidence already in hand

You need at least three. You have five, and they are unusually well-evidenced:

**1. Accuracy hides subgroup structure.** The single number 0.7767 conceals that the age effect is +0.4810 for women and −0.0322 for men — a difference significant at p = 0.0069. Two populations, one number, and the number describes neither.

**2. The threshold is a values choice, not a data-driven fact.** The sweep from t = 0.05 to t = 0.95 produces accuracy from 0.4633 to 0.7667, DP from 1.0000 to 0.5162, and EO from 1.0000 to 0.0000 — all from the *same model*. Nothing in the data picks a point on that curve. Whoever picks t is making a political decision wearing a technical costume.

**3. The metrics genuinely conflict.** At t = 0.95, PP reaches its best value (1.1136 away from parity in the generous direction) precisely where EO hits 0.0000. Moving t from 0.45 to 0.55 buys 1.0 point of accuracy for 25.82% of EO. There is no setting that satisfies all three, and the only point that achieves perfect DP and EO is t = 0.05, which approves **every single applicant** — fairness achieved by refusing to decide.

**4. The model fails confidently on cases it cannot distinguish.** 24 of 167 applicants scored above 0.75 defaulted, at a mean predicted probability of 0.8859. Zero of twenty recorded attributes separate them from the 143 who repaid, and a cross-validated model fitted inside that band reaches AUC 0.5501 — chance. The confidence is real and the knowledge is not. A single accuracy figure cannot express the difference.

**5. Hidden confounds are undetectable from performance metrics.** Dropping `credit_amount` moved duration's coefficient by 48.14% while AUC fell 0.0061 and accuracy fell 0.0033. If a confound you *know about* is invisible on the dashboard, a confound you don't know about is invisible too — by construction. This is the limitation with no remedy, and it is the strongest closing point available to you.

### For the closing

The meta-lesson the brief asks for is about transparency over single metrics. The sharpest version your data supports: the one number a policymaker wants is not merely insufficient, it is **actively misleading**, because every genuine problem you found — the subgroup reversal, the confident failures, the confounding — is invisible in it. The model's accuracy stayed within half a percentage point through all of it.

Avoid ending on "more research is needed". Say what you would actually require before deployment: which subgroup metrics get reported, who chooses the threshold and on what record, what triggers a review.

## Oral assessment prep (3 marks)

This is a fifth of your grade and it is where a report you did not write yourself falls apart. The rubric wants "excellent understanding of concepts, procedures, and outcomes; answers confidently". Work through the notebook once yourself before the viva — run it, change the seed, see what moves.

### Questions you are very likely to get

**"Why did you use linear regression on a binary outcome at all?"** Because the brief required it, and it demonstrates the problem: 12.3% of predictions fell outside \[0,1\], the largest 1.2622. A linear probability model is unbounded. Accuracy 0.6467 against logistic's 0.7767.

**"Why is your confusion matrix triangular?"** Because the two models rank applicants almost identically (Spearman 0.9903) but cut at different depths. The median rule flags exactly 50%; the 0.50 rule flags 77.7% because the base rate is 70%. A's approve-set is nested inside B's.

**"Is your Simpson's paradox result just small-sample noise?"** No — and this is why the interaction test is in there. age × female = +0.4922, p = 0.0069, 95% CI \[+0.1351, +0.8493\], LR χ²(1) = 7.6968, p = 0.0055. The raw approval rates by age band show the same crossing without any model.

**"Why did you drop `credit_amount` rather than something else?"** Because it satisfies the definition of a confounder: correlated with the retained predictors (+0.6268 with duration, −0.3062 with installment rate) *and* independently predictive of the outcome (−0.3608, p = 0.0177). Dropping a variable uncorrelated with the others would move nothing.

**"How do you know the coefficient shifts aren't just numerical noise?"** Because the OVB formula predicted them. β\_omitted × δⱼ gave the right sign for 6 of 6 predictors, and for duration it predicted −0.1974 against an observed −0.1828.

**"Why only 26 adversarial rows when the brief asks for 50?"** Because only 26 exist in a 300-row test set, and the brief says to use all of them if fewer than 50 exist. Padding would mean including cases that are not high-confidence errors.

**"Why does your model confidently get those 24 wrong?"** This is the question to be ready for. Not because of a missing interaction — I tested five, several were significant, none reduced the error count. Because 0 of 20 recorded attributes separate those 24 from the 143 correct cases (smallest p = 0.0566), and cross-validated AUC inside the band is 0.5501 against chance 0.50. The model learned a group base rate of 89% for no-checking-account applicants and applies it to everyone in the group. Aleatoric uncertainty reported as confidence.

**"Why is `checking_status = A14` positive? Shouldn't having no bank account be a bad sign?"** In training, A14 applicants were approved 89.0% of the time against 52.4% for A11 (overdrawn). The codebook's A11 means a *negative* checking balance, so having no account is better than having a visibly bad one. The model is reading a real pattern, but it is reading absence of information as a positive signal, which is worth flagging as a risk.

**"Why did you use your own logistic implementation instead of statsmodels or sklearn?"** sklearn's `LogisticRegression` applies L2 regularisation by default, which biases coefficients and invalidates the standard errors — unusable for inference. The hand-written Newton–Raphson is unpenalised MLE, the same estimator statsmodels uses, with standard errors from the inverse observed Fisher information. It also means the notebook runs with no extra dependencies.

**"What would you do differently?"** Have a real answer ready. Options grounded in what you found: collect a feature that distinguishes within the no-checking-account group; model sex and marital status separately at the data-collection stage instead of entangling them; report subgroup metrics rather than pooled accuracy; pre-register the threshold choice so it cannot be tuned after seeing group outcomes.

### Things to be able to do on the spot

- State what the 70/30 split was stratified on, and why.
- Explain what a coefficient of +1.7869 means in plain words (odds multiply by 5.97).
- Explain what "per 1 SD" means for a scaled coefficient, and convert to per-10-years for age.
- Say what DP, EO and PP each measure, and name a situation where optimising one harms another — you have t = 0.45 → 0.55 for exactly that.

## Weak points a grader could probe

Better to raise these yourself in the report than have them found. Acknowledging a limitation is worth marks; missing it is not.

**Small test subgroups.** The female test group is 85 people. At t = 0.95 only 10 women are approved, so PPV and the fairness ratios at the top of the sweep rest on very few cases. The EO value of 0.0000 at t = 0.95 occurs because female FPR hits exactly zero — a ratio with zero in the numerator, not a meaningful "infinitely unfair" reading. Say so.

**The t = 0.05 degenerate point.** DP = EO = 1.0000 there only because every applicant is approved. Do not present it as a fairness optimum; present it as the reductio it is.

**Two different model specifications.** Part 2.1 uses a 7-attribute core model; Parts 1, 2.2, 2.3 and 2.4 use the full 42-feature model. This is justified (subgroup sample sizes), but it means the age coefficient in Part 2.1 is not directly comparable to Part 2.2's table. State the reason explicitly rather than letting it look like an inconsistency.

**Full data vs training data in Part 2.1.** The Simpson analysis is fitted on all 1000 rows because it is a descriptive, structural question rather than a predictive evaluation, and subgroup power matters. The other parts respect the train/test split. Again — defensible, but say it.

**`personal_status_sex` is in the model.** Sex is a direct input to a model you then audit for sex-based fairness. That is realistic and worth discussing rather than hiding: it is one reason the disparity exists, and removing it would not eliminate the disparity because other features proxy for it.

**Percentage change on small coefficients.** Covered in the Part 2.2 section — restrict to significant predictors and report SE shifts alongside.

**The dataset is from 1994 Germany.** Currency is Deutsche Marks, the sex/marital encoding reflects the era, and nothing here should be read as a claim about lending today. One sentence somewhere in the report covers this.

**A seed is one draw.** Every number here is conditional on `random_state = 224`. The findings that are backed by significance tests (the interaction at p = 0.0069, the OVB shifts) are robust; the exact counts (83 mismatches, 26 adversarial rows) would move under a different seed. If you want extra credit, re-run with two or three seeds and report the range — the notebook makes this a one-line change.
