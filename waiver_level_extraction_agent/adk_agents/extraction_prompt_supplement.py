"""
Generic additions to the overview extraction prompt, used by the ADK path only
(appended to sub_agents.overview_prompt.OVERVIEW_AGENT_PROMPT, which is left
untouched so the CLI pipeline is unchanged).

Everything here is form-generic - nothing refers to a particular waiver. Each
rule targets a failure seen when comparing a raw run against the verified CSV:
option-group checkboxes returned as label text / null, per-year fields filled
with one repeated figure, and null used where the form shows an unchecked box.
"""

PROMPT_SUPPLEMENT = """

## ADDITIONAL EXTRACTION GUIDANCE (these refine the rules above; follow them)

A. **Option-group checkbox fields.** Many schema fields are one option from a
   "select one" / "check each that applies" group, and their field name is the
   option's label (often with no "0 = No, 1 = Yes" hint). Treat every such
   field as a binary checkbox: return 1 if that option's box is checked and 0
   if it is unchecked. NEVER return the option's label text as the value, and
   never return the label for the checked option while leaving its siblings
   null. Each option in the group is its own field and each gets 0 or 1.

B. **Unchecked is 0, not null.** If the group of options is present in the
   document and you can see its boxes, every box you can see that is not
   checked is 0. Use null only when the question or group does not appear in
   the document at all, or the document states the section is not applicable.

C. **Sections the form says to skip.** If the document says a section or
   appendix is not applicable or "do not complete" (including pages that only
   say the answers elsewhere indicate it need not be submitted), the fields
   inside that section have no source value: return null for them, do not
   invent 0 or text, and say so in the evidence ("section not applicable:
   <short quote>"). The gating question itself (e.g. an Applicability yes/no)
   is still extracted normally.

D. **Per-year and per-column fields.** Fields that differ by year (Year 1,
   Year 2, ...) or by column must be read from that year's own row/column.
   Never repeat one figure across all years, and never use a grand total or
   cumulative total for a single year. If a specific year is genuinely blank,
   return null for that year.

E. **Coded and numeric fields return the code or number only.** For fields
   described with a code legend (e.g. "0 = ..., 1 = ..."), return the integer
   code for the selected option, not the option's text. Pay attention to the
   exact wording of the legend, including inverted codings.

F. **Completeness pass.** Before you finish, go through the schema one field at
   a time. For every field you are about to leave null, search the whole
   document (all appendices, tables and attachments) for the field's label or
   obvious synonyms - the schema "Header" is only a hint, and answers often sit
   in a different appendix. Only leave a field null if it is truly absent or
   covered by rule C.
"""
