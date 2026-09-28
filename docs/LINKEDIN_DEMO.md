# Evidence-first demo rehearsal

Record a 75–90 second story in the Next.js interface. Use a fresh session for each rehearsal.
The calculation card is shown only when the response includes a backend-computed derivation
with complete links to its source claims and citations. It never invents a comparison.

## Recording sequence

1. Open with a finished chart. Click a bar, show the selected value beside its original PDF page,
   and close the viewer to reveal the retained bar selection and source link.
2. Ask: “Compare the 2011 total sex ratios of Karnataka, Odisha, and Madhya Pradesh.”
3. Show “Show the calculation” if the response includes a derived comparison. If needed, ask the
   explicit follow-up: “What is the difference between Odisha and Karnataka, in females per
   1,000 males?” Open each operand's source. The documented statewide values are 979 and 973,
   with a difference of 6; verify these against the displayed source pages during rehearsal.
4. Ask: “Create a bar chart of the 2011 total sex ratios of those three states.” Click a bar,
   inspect the quoted row and original page, then close the viewer.
5. Ask: “Which district in Karnataka had the highest total literacy rate in 2011?” Inspect the
   answer and citation. Include this only if it fits the recording time.
6. Ask: “What were the corresponding sex ratios in 2021?” Show the actual unsupported-answer
   response. The supplied 2011 reports do not establish those figures.
7. End on the chart with its selected-source link and the repository URL.

Opening narration: “This chart came from a census PDF. Click any bar, and you can inspect the
evidence behind it.”

## Before recording

- Run `make check` and `make web-check`.
- Rehearse the sequence three times against the actual stack. Record outcomes and latency;
  resolve failures before recording. These requests call Vertex AI and incur model usage.
- Check source links, calculation operand order, units, keyboard activation, Escape to close,
  and selection persistence after closing the viewer.
- Use large text, hide unrelated tabs and notifications, and add captions.
- If waiting periods are shortened, mark them “wait shortened.” Progress reflects real graph
  events; there is no simulated completion percentage or prerecorded answer in the product.
- When PDF highlighting is unavailable, keep the explanation visible and compare the quote
  with the page directly. Do not imply that an exact cell was highlighted.

Offline checks and fixture-based UI validation are not evidence that the live sequence passed.
Record live rehearsal results separately with the date and model used.
