# CWRU-Like Dataset Discovery Report

## Scope Decision

This release follows the CWRU-only minimal-project option in the course manual §19.
The paper therefore makes claims only about the selected CWRU protocol and does not
claim cross-dataset generalization.

## Candidates Considered

The sources below were checked on 2026-07-23. The URLs are recorded so that the
search can be repeated without relying on an undocumented web query.

| Dataset | Official or primary source | Access / terms found | Suitability | Reason deferred |
|---|---|---|---|---|
| Paderborn University Bearing Data Center | https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter | Direct download; the site states CC BY-NC 4.0 and permits academic, non-commercial use with citation | Motor-current and vibration measurements make it a strong candidate for natural/artificial-damage transfer tests | Label taxonomy, sampling setup, and operating-condition mapping require a separately preregistered protocol; adding it after inspecting CWRU results would expand the fixed study |
| IMS Bearings | https://catalog.data.gov/dataset/ims-bearings | Public NASA Prognostics Center repository entry with downloadable files; the catalog identifies the source as IMS, University of Cincinnati | Run-to-failure vibration is useful for degradation and prognostic evaluation | Run-to-failure objectives and labels do not map directly to the four-class CWRU classification protocol |
| FEMTO-ST / PRONOSTIA | https://publiweb.femto-st.fr/tntnet/entries/1528/documents/author/data | Primary publication record describing the PRONOSTIA experimental platform and the IEEE PHM 2012 challenge data; reuse terms must be checked before redistribution | Accelerated bearing-degradation trajectories are suitable for remaining-useful-life research | The task is prognostic rather than the controlled fault-classification interaction studied here |

## Release Boundary

No candidate was downloaded, tuned, or tested for this release. Source discovery
does not count as experimental use. A future external validation should first
confirm the applicable license or challenge terms and preregister:

1. label harmonization;
2. sensor and sampling-rate mapping;
3. recording-level split units;
4. augmentation parameters fixed without reference to external test results; and
5. transfer metrics and failure conditions.

This explicit deferral prevents the single-dataset result from being presented as
industrial or cross-dataset evidence.
