# Local and Server Data-Copy Audit

Audit date: 2026-07-23

## Copies inspected

| Copy | Scope found | Selected 40-file subset | Decision |
|---|---:|---|---|
| `E:\CWRU_Bearing_Data\CWRU_Bearing_Data\cwru_experiment\Source_Datasets` | 161 `.mat` files | 2 selected files unreadable | Do not use for this experiment |
| `D:\CWRU_Bearing_Data\CWRU_Bearing_Data\Source_Datasets` | 161 `.mat` files | All 40 selected files readable | Valid local source |
| `/root/cwru-working-paper/data_manifest/raw` on the A10 server | 161 `.mat` files | All 40 selected files readable | Source used for audited rerun |

The server-generated manifest was compared by `recording_id` with the D-drive
manifest: all 40 SHA-256 hashes match, with zero missing or differing selected
files.

## E-drive corruption finding

The E-drive copy contains 13 unreadable `.mat` files in total. Two belong to
the project's exact 40-recording selection:

- `Normal_3HP_1730rpm.mat`
- `IR007_1HP_1772rpm.mat`

For the other 38 selected recordings, the D- and E-drive SHA-256 hashes match.
The two unreadable E-drive files differ from the readable D-drive versions and
appear truncated. Therefore the E-drive tree is retained only as a damaged
copy, not as experimental input.

The D-drive tree also contains 11 unreadable files outside the fixed 40-file
selection, principally from unused 48 kHz material. The preprocessing code
selects the documented 12 kHz drive-end subset first and then fails fast if
any selected file is unreadable; it never silently substitutes or skips a
selected recording.

## Reproducibility evidence

`dataset_file_manifest.json` records the SHA-256 hash, byte size, readability,
DE-channel sample count, and window count for every selected recording.
`recording_split_assignments.json` records the exact recording-level
assignment for all five seeds. The audited rerun regenerates both files before
training and exits on any mismatch in the selected data.
