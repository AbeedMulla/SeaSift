# Draft: guidelines for using AI in fish video surveys

*A starting point for Coastal Action, written by the seaSEArch team for RBC HubHacks. Not legal or policy advice. Edit freely.*

Coastal Action's brief says AI use is heavily restricted for environmental reasons, and that guidelines are still being developed. This page suggests a few simple rules, and how seaSEArch already follows them.

## 1. No AI by default
Use AI only where it clearly saves staff time that simpler methods can't.
*seaSEArch:* finding moments, scoring footage quality, enhancing pictures, the species cards and the new-arrival alerts all work with no AI model.

## 2. Only on our own computers
Only open-source models running on Coastal Action's own computers. No cloud AI services, and footage is never uploaded.
*seaSEArch:* runs offline on a laptop. The app only opens on that computer.

## 3. Smallest model first
Try the smallest model that does the job. Only move to a bigger one if the small one isn't good enough on our own footage.
*seaSEArch:* we tried the smaller original BioCLIP first. On a real survey video it named 0 to 2 of 8 crabs and shrimp correctly, and BioCLIP 2 named all 8, so the optional AI now uses BioCLIP 2. The smaller one is one setting away (`USE_BIOCLIP_V1` in `seasearch/config.py`).

## 4. People decide
AI only suggests. A person confirms every species and every count before it goes in a report.
*seaSEArch:* every species and count in the report was confirmed by a reviewer.

## 5. Only where it's needed
Run AI on the short moments that need it, not on hours of empty footage.
*seaSEArch:* the AI only looks at moments the motion check already found.

## 6. Measure it
For every survey, log how much energy the AI used, and how often its suggestions were right.
*seaSEArch:* logs an energy estimate for every video. Comparing AI suggestions with what reviewers confirmed is a next step.

## 7. Review once a year
Keep using AI only if it saves real time and is accurate enough on our own footage. Otherwise switch it off.

## 8. Protect sensitive information
Locations of species at risk are not shared publicly. Footage and species lists stay on our computers unless the team decides otherwise.
