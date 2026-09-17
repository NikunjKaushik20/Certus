# Reader study: is the explanation useful, and does review fit in 30 seconds?

The problem statement asks for Grad-CAM output "rated as clinically useful" and ophthalmologist
validation "in under 30 seconds". Neither can be measured without ophthalmologists. This folder
is everything needed to measure both in one sitting; the missing part is the readers.

## Design

- **Readers.** At least two ophthalmologists (or trained DR graders), working independently.
- **Cases.** 50 held-out eyes from Indian cameras (APTOS 2019, IDRiD), picked by
  `select_cases.py`: 25 the model auto-referred, 25 it abstained on, drawn at random with a fixed
  seed. These are the only eyes a reviewer ever sees in deployment. 28 of the 50 are referable by
  the dataset's own reference grade, so both kinds of error are possible.
- **Two arms, crossed over.** Each reader grades every case twice, at least two weeks apart, half
  the readers starting with each arm:
  - *Unassisted (A):* the photograph only (`images/<case>.jpg`). Record ICDR grade and seconds on
    `unassisted_template.csv`.
  - *Assisted (B):* the Certus review queue, with grade, lesion evidence and the precomputed
    Grad-CAM. The app times each case itself (from opening the case to submitting), and the
    reader fills in the three optional ratings under the notes box:
    Grad-CAM usefulness (1 misleading … 5 decisive), lesion-evidence usefulness (same scale), and
    whether the heatmap sits on the lesions that justify the grade (yes / partly / no).

## What gets reported (`analyse.py`)

| Question | Measure |
|---|---|
| Does review fit in 30 s? | median and 90th-percentile seconds per case, share of cases ≤ 30 s, per arm |
| Is the explanation useful? | distribution of the 1–5 ratings; share rated 4 or 5 |
| Does the heatmap point at the right thing? | share of yes / partly / no |
| Does assistance cost accuracy? | referral sensitivity and specificity (grade ≥ 2) and exact-grade agreement against the reference, per arm |

Pre-registered success criteria, fixed before anyone reads: median assisted time ≤ 30 s, at
least 70% of Grad-CAM ratings 4 or 5, and assisted referral sensitivity no lower than unassisted.

## Running it

```bash
python study/select_cases.py            # writes cases.csv and images/ (already done; seed fixed)
# start the API and the web console (see the top-level README), then:
python study/load_cases.py              # 50 visits, graded, each landing in the review queue
# readers use the Review page (arm B) and the images folder + CSV (arm A)
python study/analyse.py                 # -> study/results.json
```

Ratings are stored as `review.explanation_rated` events in the hash-chained audit log, so they
cannot be edited after the fact without the chain showing it.

## Limits

Fifty cases and two readers give wide intervals: a proportion of 0.8 on 100 ratings has a 95%
interval of roughly ±0.08. It is a usability study, not a diagnostic-accuracy trial. The cases are
retrospective photographs, not a live camp.
