# LendingClub — Current Loans Analysis

Exploratory data analysis on the LendingClub loan dataset, testing whether the *purpose* of a loan predicts whether it stays "Current" (actively being paid on time).

This was my section of a group project on loan evaluation, covering the "Current loans, not late on any payments" slice of the data.

## Hypothesis

Loans issued for debt payments (debt consolidation + credit card refinancing) are more likely to stay Current than loans issued for investment (small_business as proxy) or educational purposes.

## Finding

Debt-payment loans stay Current at a meaningfully higher rate (82.0%) than small-business loans (76.3%) — a 5.7 percentage-point gap, statistically significant.

![Success rate by purpose group](figures/success_rate_by_purpose.png)

## Key statistics

| Test | Result |
|---|---|
| Chi-square | 120.79 |
| p-value | 5.91 × 10⁻²⁷ |
| Degrees of freedom | 2 |
| Cramér's V | 0.015 |
| z-test (debt vs investment) | z = 10.98, p < 0.001 |

The chi-square confirms purpose and Current-status are not independent. Cramér's V is small in absolute terms, which is expected — purpose is one of many factors influencing loan outcomes, but the effect is consistent and meaningful at the portfolio level.

## Methodology

- **Data**: LendingClub public dataset, 887,379 loans
- **Filter**: Loans issued in 2014–2015 (≈556k loans). This controls for "Current" being a point-in-time status — older loans that have already finished cannot show as Current regardless of how well they performed
- **Grouping**:
  - `debt_payments` = debt_consolidation + credit_card
  - `investment` = small_business
  - `educational` = educational (dropped from interpretation; only 1 loan in window)
- **Tests**:
  - Wilson 95% confidence intervals on each success rate
  - Chi-square test of independence with Cramér's V for effect size
  - Two-proportion z-tests for pairwise comparisons

## Supporting finding

60-month loans show a higher Current rate (83.7%) than 36-month loans (81.0%). This is *not* evidence that longer loans are safer — it is a survivorship effect. 36-month loans from 2014–2015 had time to finish and move to "Fully Paid," while 60-month loans were still actively paying when the data was pulled.

![Success rate by loan term](figures/success_rate_by_term.png)

## Repo structure

```
lendingclub-current-loans/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── analysis.py            # full OOP analysis pipeline
├── figures/
│   ├── success_rate_by_purpose.png
│   └── success_rate_by_term.png
├── report/
│   └── report.md              # written report draft
└── data/
    └── README.md              # dataset instructions (CSV not tracked)
```

## How to run

1. Download `LendingClub_Data.csv` and place it in the `data/` folder (see `data/README.md`).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run:
   ```
   python src/analysis.py
   ```

The script prints summary tables and test statistics to the console, saves both figures into `figures/`, and writes a cleaned analysis CSV (`lc_analysis.csv`) into the figures folder alongside the plots.

## Tech stack

Python 3 · pandas · numpy · scipy.stats · statsmodels · matplotlib · seaborn

## Author

Alexandre Lenfers — Data Science, University of Arkansas
