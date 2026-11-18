# Labelling guide

This guide tells a labeller how to fill the label sheets. The sheets measure how well the
`misinfo` model and the Laya model work on real corpus records. Your labels are the
ground truth for that measurement. Read the full guide before you label the first row.

## The task

Each row is one corpus record. You give it two labels.

| column | question | values |
|---|---|---|
| `label_a` or `label_b` | Is the record misinformation-like? | `1` yes, `0` no |
| `reports_on_false_claim_a` or `_b` | Does the record report on or rebut a false claim? | `1` yes, `0` no |

Leave a cell blank if you cannot decide. A blank cell is not the same as `0`. Use the
`notes` column to say why you cannot decide.

Do not use any other value in a label cell. The tools count a different value as invalid.

## What you label

On the English sheet, label the `text` column. The `title` column is context.

On the Hindi sheet, label the `title` column. The Laya model reads only the title.
The `text` column is context. Do not change a label because of a fact that only the
text gives.

Do not search the web for the truth. Judge the record on what it says and on what you
know. If a claim needs a search to check, put `0` in `label_a` or `label_b` and write
"unchecked" in `notes`. Do not guess.

Ignore the `stratum` and `weight` columns while you label. They are codes for the
sample draw and they must not change your judgement.

## What is misinformation-like

Label `1` when the record states or implies a factual claim that is false or fabricated.

Label `1` in these cases:

- The record presents a real photo, video, or document as something it is not.
  Example: an old video shown as a new event.
- A fact-checker rated the claim false.
- The record invents a quote, a statistic, or an event.
- The record presents a cure, a threat, or a rumor as fact with no support, and the claim
  is false by common knowledge.
- The record asks the reader to share a claim that is false.

Label `0` when none of these cases applies.

Label `0` in these cases:

- The record is an opinion, a joke, satire that is clearly marked, or a question.
- The record states no claim that you can check.
- The record reports a true event, even if the tone is angry or the topic is political.
- The record is about a topic you dislike, but the claims in it are true or are not
  checkable.

Misinformation-like is a judgement about the claim in the record. It is not a judgement
about the author or about the outlet. Do not label a record `1` because of who wrote it.

## Records that report on a false claim

Many news articles name a false claim in order to correct it. A fact-check article is the
clearest example. A news report on a viral hoax is another example.

These records repeat the false claim, but they do not push it. A model can mistake them for
misinformation. The sheet measures this with a separate column.

Fill the two columns in this way:

1. Set `reports_on_false_claim` to `1` when the record names a false claim and the
   record corrects it, rebuts it, or reports that it spreads.
2. Set `label` to `0` for such a record, if the record does not also push the claim.
3. Set `label` to `1` for such a record only if the record states the false claim as true
   in some part. Example: a headline that repeats the claim as fact and a rebuttal
   in the last line.
4. Set `reports_on_false_claim` to `0` for a record that pushes a false claim with no
   correction. Set it to `0` for a record that has no false claim.

The two columns are independent. A record can have `label` `0` and `reports_on_false_claim`
`1`. This is the normal case for a debunk. A record with `label` `1` and
`reports_on_false_claim` `1` is rare. Explain it in `notes`.

## Edge cases

| case | `label` | `reports_on_false_claim` |
|---|---|---|
| Fact-check article that rebuts a claim | `0` | `1` |
| News report: "a viral post claims X, police say it is false" | `0` | `1` |
| Post that repeats a claim and adds "this is fake" | `0` | `1` |
| Post that repeats a claim and says "share before it is deleted" | `1` | `0` |
| Quote of a false claim with no comment | `1` | `0` |
| Question: "Is X true?" with no answer | `0` | `0` |
| Clearly marked satire | `0` | `0` |
| Old true news shown as new, with the date missing | `1` | `0` |
| Claim that you cannot check | `0` and note "unchecked" | `0` |
| Video record with only a title and a short description | Label what the record says | `0` or `1` |
| Link or hashtag only, with no claim | `0` | `0` |
| Text in another language than the sheet | Leave blank, note "language" | blank |

If a record fits two rows of the table, use the row that is more specific. If you are
still not sure, leave the cell blank and write a note. The adjudicator decides.

## Hindi notes

- Label the title, as the section "What you label" says.
- A Hindi title can be a short headline that names a claim in few words. Label the claim
  that the title makes. Do not add facts from the article.
- Hindi headlines often end with a question mark to hint at a claim. Label `1` if the title
  clearly implies the claim is true. Label `0` if it is a real open question.
- Words such as "वायरल" (viral), "दावा" (claim), and "सच" (truth) show that the record
  reports on a claim. They do not show that the claim is false. Read the rest of the title.
- A record in Hinglish (Hindi in Latin letters) is in scope. Label it in the same way.
- A record in a mix of Hindi and English is in scope. Label it in the same way.
- A record in a third language, such as Urdu or Punjabi, is out of scope. Leave the cells
  blank and write "language" in `notes`.
- Both labellers on the Hindi sheet must read Hindi. A labeller who needs a translation
  tool for most rows must not label the Hindi sheet.

## How the labelling works

Two people label each row, and they work alone. Do not look at the other labeller's cells.

Do not look at a model score. If you see a score, tell the lead and stop.

1. Labeller A fills `label_a`, `reports_on_false_claim_a`, and `labeller_a` (your initials).
2. Labeller B fills the `_b` columns on a separate copy of the sheet. The lead merges the
   two copies into one sheet.
3. Run the agreement report:
   `python -m modeling.cli label-agreement <sheet>`. It prints Cohen's kappa with an
   interval, the raw agreement, and the ids where the two labellers differ.
4. Run the same command with `--fill-agreed`. It copies each shared label into
   `adjudicated_label` and `adjudicated_reports_on_false_claim`. It leaves the rows where
   the labellers differ blank.
5. A third person, or both labellers together, settle each blank row. Write the final
   value in the adjudicated columns. Explain the decision in `notes`.
6. Set `labeller_kind` to `human` on every labelled row. Use `model-unreviewed` only
   for labels that a model made and that no person read. The evaluation commands refuse
   such rows unless the caller passes `--allow-model-labels`. Every output then says
   `labeller: model, unreviewed by a person`.

Run the evaluation with `make relabel-eval`. It rewrites the agreement reports and the
transfer metrics from the sheets in one run.

## Keep the sheets private

The sheets hold corpus text. Keep them outside git. Git ignores the files in
`artifacts/hand_labels/`. Do not paste row text into an issue, a pull request, or a chat.
Ids are safe to share.
