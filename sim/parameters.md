# Capacity-model parameters and where they come from

Defaults in `capacity.py`. Every value is either sourced or marked as an assumption to sweep.

| Parameter | Default | Source / status |
|---|---|---|
| Annual screening target (district) | 100,000 patients | Problem statement SIH 26038 |
| Ungradable images, field conditions, **no** quality handling | 22.5% | SMART India: 6,133 of 7,910 people with diabetes (77.5%) had gradable images — Lancet Global Health 2022 |
| Ungradable images, Thailand AI deployment (clinic lighting) | 21% | Beede et al., CHI 2020 (≈1,840 images) |
| Ungradable images, handheld camera with trained operator | ~4% | Remidio Fundus-on-Phone primary-care study (10/261) |
| DR prevalence among people with diabetes | 12.5% | SMART India |
| Vision-threatening DR prevalence | 4.0% | SMART India |
| Referable DR share used in the model | 8% | **Assumption** between VTDR (4%) and any DR (12.5%); swept 4–12.5% |
| Referral non-attendance for STDR | ~90% | SMART India follow-up (qualitative study, PLOS One 2022) — motivates same-day reporting |
| Rural power supply | 22.6 h/day average | Ministry of Power / PIB, Feb 2025 |
| Outage frequency | ≥ 1/day for two-thirds of rural households | CEEW / Prayas surveys |
| Patients per 2-hour session with cloud upload | ~10 | Beede et al. (upload delays were a bottleneck) |
| Capture time per patient (both eyes) | 5 min mean (lognormal) | **Assumption** (non-mydriatic, 2 fields); swept 3–8 min |
| Ophthalmologist time per case, manual grading | 90 s | **Assumption**; published annotation times range from tens of seconds (healthy) to much longer (severe); swept 45–180 s |
| Ophthalmologist time per case with Certus report | 30 s | Problem-statement target (< 30 s validation) |
| Consent (ABDM OTP) step | 2 min mean, 5% failure → retry | **Assumption** |
| Edge inference | 3 s | **Assumption**, to be replaced by measured GPU/Jetson timing |
| Reviewer tele-review hours | 4 h/day, 5 days/week | **Assumption** (ophthalmologists also run clinics) |

## Sources
- Lancet Global Health 2022, SMART India: https://www.thelancet.com/journals/langlo/article/PIIS2214-109X(22)00411-9/fulltext
- Beede et al., CHI 2020: https://dl.acm.org/doi/10.1145/3313831.3376718
- Remidio handheld primary-care study: https://pmc.ncbi.nlm.nih.gov/articles/PMC13186558/
- STDR treatment-seeking (PLOS One): https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0270562
- PIB, rural power supply 2025: https://www.pib.gov.in/PressReleasePage.aspx?PRID=2105394
- CEEW electricity access: https://www.ceew.in/publications/access-to-electricity-availability-and-electrification-percentage-in-india
- Prayas, quality of supply: https://energy.prayaspune.org/our-work/article-and-blog/quality-of-electricity-supply
