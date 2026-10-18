# Evaluation

Measures how accurately the pipeline extracts fields, and whether it routes
the right documents to human review, on a labeled dataset that covers plain
text, text-layer PDFs and scanned PDFs.

Results for each run are in [results/README.md](results/README.md).

## Dataset

56 synthetic documents in [data/](data/), built by `build_dataset.py` from a
fixed seed: a main set of 44 and 12 held out. Running it again writes byte-identical files. All names,
companies, addresses and 555 phone numbers are fictional.

| Type | Count | Missing one required field |
|---|---|---|
| Invoice | 12 | 3: due date, invoice number, currency |
| Resume | 9 | 2: phone, summary |
| Contract | 9 | 2: expiration date, governing law |
| Other (recipe, meeting notes, building notice) | 3 | n/a |
| Poor-scan copies of the 11 scanned documents | 11 | the same as the originals |

Each type is split evenly across three formats: plain text, PDF with a text
layer, and scanned PDF. Scans are rendered at 115 DPI, slightly rotated,
blurred and JPEG-compressed, with no text layer, so only the vision model can
read them.

The 11 scanned documents appear a second time as poor scans: 80 DPI, up to 2°
of skew, heavier blur and JPEG quality 30, about what a low-resolution fax or
a hurried phone photo gives. A person can still read them, but 3s, 6s and 8s
start to look alike. The labels are the same as the originals', so the gap
between the Scanned PDF and Poor scan columns is the effect of scan quality
alone.

A document missing a required field should be sent to human review, and so
should every document of no supported type. Everything else should be stored
without review.

**Held out:** 12 more documents (4 invoices, 4 resumes, 4 contracts), poor
scans only, all complete. They were added after the second read was chosen
on the main set's poor scans, so they measure it on documents it was never
tuned on. Results report them separately. Resumes reuse the same nine names,
with different contact details, skills and jobs.

Contracts are service, consulting, license, lease and supply agreements, all
of which carry payment terms. NDAs are left out because the pipeline treats
payment terms as required for every contract (see Known Limitations in the
main README).

## Scoring

`scoring.py` compares each ground-truth field with what the pipeline
extracted. Formatting differences don't count as errors; value differences
always do:

- dates match in any common printed format
- amounts match within one cent
- company names match without suffixes like LLC or Inc.
- phone numbers match on digits
- skills match when the overlap F1 is at least 0.8
- line items match on count and per-line totals
- experience, education and contract parties match as sets of names
- free-text fields (summary, key terms, payment terms, termination clause)
  only need to be present. Judging their wording would need a second model as
  a judge, and that model's errors would end up in these numbers.

A value for a field the document doesn't have counts as an error, even a
placeholder like "N/A". The confidence gate counts any value as present, so a
placeholder gets past it the same way an invented value does.

## Metrics

| Metric | Meaning |
|---|---|
| Fields correct | Share of all ground-truth fields scored correct, across every document of a supported type |
| Absent fields left empty | Of the deliberately missing fields, how many the model left empty instead of filling in |
| Auto-stored | Documents stored without human review |
| Silent errors | Auto-stored documents with any wrong field, a wrong type, or a missing field that should have sent them to review. These are the costly ones: nobody looks at them. The results page lists each one and what was wrong |
| Auto-store precision | Share of auto-stored documents that were entirely correct |
| Straight-through | Of the documents that should be stored without review, how many were |
| Review catch rate | Of the documents that should go to review, how many did |
| Cost per 1,000 docs | From the API's token counts and the prices in `pricing.py`, both reads included |
| Pages flagged by check | Vision pages where the second read disagreed with the first (and, in the JSON, pages left unchecked because the second read failed) |
| Errors | Documents where the pipeline raised an exception. Their fields count as not extracted and the tokens they used still count, so a crash lowers the numbers instead of dropping out of them |

Every document is sent under a neutral file name (`document.pdf`), since the
dataset's own file names would tell the classifier the answer.

## Running it

```bash
python -m evals.build_dataset          # only needed after changing the generator
python -m evals.run_eval --model gpt-4o-mini --model gpt-6-luna
python -m evals.run_eval --model gpt-6-luna --reasoning-effort low
python -m evals.run_eval --model gpt-6-luna --check-model off   # without the second read
```

Runs use the app's second-read model (gpt-6.1-sol) unless `--check-model`
says otherwise. Each run makes real API calls. A full run over 56 documents
costs well under a dollar on the small models.

## How the second read was chosen

Before building it, every scanned page in the main set (11 ordinary, 11
poor) was read several ways and each read was compared word for word with
the text actually printed. gpt-6-luna misread 8 of the 11 poor scans and
none of the ordinary ones. Each candidate check was scored on whether it
flagged those 8:

| Check | Misreads caught (of 8) | False alarms | Extra cost per scanned page |
|---|---|---|---|
| gpt-6-luna reads the page twice | 4 | 1 | $0.0002 |
| gpt-6-luna again at reasoning effort medium | 4 | 0 | $0.0006 |
| gpt-6.1-sol reads it; any disagreement goes to review | 8 | 0 | $0.0073 |
| every page under 100 DPI goes to review | 8 | 3 | $0 |
| ask the model to mark characters it's unsure of | 0 | 0 | about $0 |

The same model repeats its own misreads (2026 read as 2025 both times),
and no model marked anything as uncertain. The 100 DPI cutoff only looks
good because it splits this dataset's two scan qualities; the usual 150 DPI
rule would send every scan to review. The gpt-6.1-sol read was chosen, and
the 12 held-out poor scans were added to measure it fairly. In the full
runs it cost about $0.0076 per page, in line with the experiment. An
earlier round of runs showed about $0.003; those requests were most likely
served from OpenAI's prompt cache, since the same pages had been read hours
before, and a new document never is.

The script and every read are in [experiments/](experiments/);
`python -m evals.experiments.second_read --summary` prints this table from
the saved reads without calling the API.

## Limits

- **Small and synthetic.** 56 documents from templates the pipeline's author
  wrote. Use the numbers to compare models and changes, not as a production
  accuracy figure. Real documents have layouts, languages and handwriting
  these templates don't.
- **Simulated scans.** Skew, blur and compression applied to a clean render.
  Real bad scans also have shadows, folds, stamps and handwritten notes.
- **One run per configuration.** Model output varies a little from run to
  run, so a difference of one or two documents is within noise.
- **No reviewer in the loop.** Documents that pause for review aren't
  resumed, so corrections aren't measured.
