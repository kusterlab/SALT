# SALT

A mass-spectrometry analysis pipeline for identifying the **RNA moiety of peptide-RNA crosslinks
(Sequential Analysis of peptide-xLinked nucleoTides)**.

It takes MSFragger-calibrated mzML files plus PSMs, deconvolves paired RNA MS/MS, matches the 
observed RNA ladder peaks against a theoretical fragment library, scores candidate sequences 
with a Hyperscore variant (Fenyö & Beavis 2003), and assigns significance via a non-canonical-nucleotide decoy FDR. 
It also reports a per-scan delta score and an E-value (Fenyö & Beavis 2003) as a reference metric for users.

## Prerequisites

### ProteoWizard (msconvert)

The `calibrated_mzml_transfer` step requires **ProteoWizard msconvert** to convert Thermo RAW
files to mzML, which must be installed separately. Reading Thermo RAW files needs the vendor
libraries, which are Windows-only – so the step picks its backend from the operating system:

| OS | Backend | What you must install |
| --- | --- | --- |
| Windows | native `msconvert.exe` | ProteoWizard (vendor-file build) |
| Linux / macOS | ProteoWizard Docker image (msconvert under Wine) | Docker |

#### Windows – native msconvert

1. Download the latest **ProteoWizard** installer from
   <https://proteowizard.sourceforge.io/download.html> (choose the version with vendor-file support).
2. Install it (default path: `C:\Program Files\ProteoWizard\`).
3. Set `msconvert_exe` in your `config.yml` to the full path of `msconvert.exe`
   (e.g. `C:/Program Files/ProteoWizard/msconvert.exe`).

#### Linux / macOS – msconvert via Docker

There is no native msconvert on Linux or macOS, so the step runs the official ProteoWizard
container instead. `msconvert_exe` in your `config.yml` is ignored on these platforms and 
can be left blank.

1. Install **Docker** ([Engine](https://docs.docker.com/engine/install/) on Linux,
   [Desktop](https://docs.docker.com/desktop/install/mac-install/) on macOS) and make 
   sure the `docker` command works for your user without `sudo` (on Linux, add yourself to the `docker` group).
2. Pull the ProteoWizard image:

   ```bash
   docker pull chambm/pwiz-skyline-i-agree-to-the-vendor-licenses
   ```

   The image name embeds acceptance of the vendor licenses – by using it you agree to the
   Thermo/vendor terms bundled in ProteoWizard.
3. Run the pipeline as usual – the step starts a container per RAW file on its own. It
   bind-mounts your RAW folder and mzML output folder into the container, so both must be on
   the local filesystem and visible to the Docker daemon (on macOS, listed under *Docker
   Desktop → Settings → Resources → File sharing*). Network shares will not work.


### FragPipe (peptide search)

The pipeline starts from a completed **FragPipe/MSFragger** offset search – it does not run the
peptide search itself. Two of its outputs are required inputs here: the `psm.tsv` per experiment
and the **`*_calibrated.mzML`** files (see [Required inputs](#required-inputs)).

We recommend running the search with the workflow shipped in this package:

```
salt/data/Dec2025_offset_4SU_7nt_labile.workflow
```

Load it in FragPipe via *Workflow → Load workflow → Custom*, then set the FASTA path
(`database.db-path`) to your own database. 
The workflow is tuned for 4SU peptide–RNA cross-links and configures, among others:

| Setting | Value | Why it matters here |
| --- | --- | --- |
| `mass_offsets` | 5538 offsets, up to 7 nt | The RNA adduct masses searched as precursor offsets |
| `labile_search_mode` | `labile` | Treats the RNA adduct as labile, as expected for SALT |
| `localize_delta_mass` | `true` | Localizes the cross-link to a residue |
| `calibrate_mass` | `2` | **Produces the `*_calibrated.mzML` files this pipeline consumes** |


`calibrate_mass=2` is the setting to preserve if you adapt the workflow: without mass calibration
FragPipe writes no `*_calibrated.mzML`, and `calibrated_mzml_transfer` has nothing to transfer
isotope patterns onto.

Match `preprocessing.peptide_seq_frag` and `preprocessing.peptide_seq_ce` in your `config.yml` to
the fragmentation method and collision energy actually used for the **peptide** search.

## Installation


**Option A: exact development environment.** [`SALT_env.yml`](SALT_env.yml) pins the
versions behind the published results (Python 3.14, pyopenms 3.5.0, pandas 3.0.2, …).

```bash
conda env create -f SALT_env.yml
conda activate SALT
pip install -e .
```

**Option B** Use this if Option A fails: exact pins on Python 3.14 do not
resolve on every platform. Installs from [`pyproject.toml`](pyproject.toml), whose lower bounds
(`numpy>=1.24`, `pyopenms>=3.0`, …) let pip pick versions that fit your interpreter. Any Python
`>=3.11` works; 3.12 has the widest pyopenms wheel coverage. `venv` works instead of conda.

```bash
conda create -n SALT python=3.12
conda activate SALT
pip install -e .
```

This installs everything the pipeline needs, including plotting. Add `".[dev]"` if you want to run
the tests.

Before running, copy `config.example.yml` to `config.yml` and edit the paths inside it (see
[Configuration](#configuration) below).

## Running the pipeline

The pipeline is a sequence of standalone steps orchestrated by a runner. Run it as a module
from the directory containing your `config.yml` (e.g. the repo root):

```bash
python -m salt
```

There are no command-line options. Everything is configured in `config.yml` (run with `--help`
to print usage).

The list of steps and their filenames is read from `config.yml` under the `pipeline:` section.
`config.yml` is the single source of truth for all configuration, including `max_workers`, which
sets how many parallel-safe steps run at once. The runner locates `config.yml` by walking up from
the current directory.

### Plotting matched spectra

[`salt/matched_scan_plotting.py`](salt/matched_scan_plotting.py) draws the annotated
spectra for matched scans. Run it manually **after** the pipeline finishes:

```bash
python -m salt.matched_scan_plotting
```

It writes one PDF per experiment to `analysis_output_dir`
(`{experiment}_RSM_matching_scores_spectra.pdf`) and reads the same `config.yml` as the pipeline,
so it needs no arguments.

It is deliberately **not** a pipeline step. The block of `TEST_*` constants at the top of the
script selects what gets plotted – one manifest row, a list of scan IDs from a file, a row cap, an
intensity cutoff, a custom output suffix – so you can plot every spectrum or just the handful you
care about. Edit those constants in the script and re-run; with all of them left at `None` it plots
every experiment and every matched scan.

## Configuration

Copy [`config.example.yml`](config.example.yml) to `config.yml` and edit it for your data. The
settings you most often need to change:

| Key | What it sets |
| --- | --- |
| `input_dir` | Folder with your RAW files and the FragPipe `.fp-manifest` |
| `analysis_output_dir` | Where pipeline outputs are written |
| `search_folder` | FragPipe search results (holds `{experiment}/psm.tsv`), under `input_dir` |
| `manifest_input_file` | FragPipe manifest filename (`.fp-manifest`), under `input_dir` |
| `msconvert_exe` | Path to the ProteoWizard `msconvert` executable (Windows only; leave blank on Linux/macOS, where msconvert runs via Docker) |
| `matching.theoretical_spectra_path` | Theoretical fragment library to match against |
| `tolerance.value` / `tolerance.unit` | Fragment peak-matching tolerance (e.g. `0.01` / `da`) |
| `max_workers` | How many parallel-safe steps run at once (`1` = serial, safest) |
| `FDR_control.rsm_fdr_level` | Decoy-based FDR threshold (e.g. `0.01` = 1%) |

These are the common entry points; `config.example.yml` is commented and lists the full set of
options (calibration, deisotoper parameters, intensity filtering, plotting, etc.).

### Required inputs

Under `input_dir`, the pipeline expects:

- your Thermo **RAW files**,
- the **`{rawfile}_calibrated.mzML`** files written by FragPipe's mass calibration
  (`calibrate_mass=2`) – one per RAW file,
- the FragPipe **`.fp-manifest`** (`manifest_input_file`), and
- FragPipe **search results** under `search_folder`, one subfolder per experiment, each
  containing a `psm.tsv` (i.e. `{search_folder}/{experiment}/psm.tsv`).

The `*_calibrated.mzML` files must sit **next to their RAW file** – the pipeline resolves each one
as a sibling of the `rawfile_path` recorded in the manifest, not by searching `input_dir`. If FragPipe
writes them somewhere else, copy or move them alongside the RAW files before
running. 

#### Why msconvert is still needed when a `_calibrated.mzML` already exists

FragPipe's calibrated mzML is mass-calibrated but **deisotoped**. 
The RNA ladder deconvolution downstream needs the **full isotope
envelope**, which the calibrated file no longer contains.

So `calibrated_mzml_transfer` needs both: msconvert re-converts the RAW to get the full peak list,
and the calibrated file supplies the m/z calibration. The merged result,
`{rawfile}_calibrated_transferred.mzML`, is calibrated *and* keeps its isotope envelopes. See the
docstring of [`salt/calibrated_mzml_transfer.py`](salt/calibrated_mzml_transfer.py) for
how the transfer works.

### Regenerating the theoretical-spectra libraries

`matching.theoretical_spectra_path` selects one of the libraries in `salt/data/`. You
only need to rebuild them when the chemistry changes – a different crosslinked nucleotide, or a change
to the fragmentation method. Generate your own fragment annotations following the same format as `massdiff_adduct_annot_10nt_original.csv`, then run the scripts in [`prepare/`](prepare/), chained by
`build_libraries.py`:

```bash
python -m prepare.build_libraries --raw-adduct-table /path/to/massdiff_adduct_annot_10nt_original.csv
```

Run it **from the repo root**. It runs four steps in order –
build the ladder mass tables, add the diagnostic-ion rows, generate the target plus the
methyl/fluoro/azido decoy libraries, then copy the results into `salt/data/`. Use `--from` /
`--only` to run part of the chain, and `--no-deploy` to generate without overwriting the shipped
libraries. `--help` lists the steps.

This is a **manual, occasional** task and is deliberately not part of `python -m salt`.


## Outputs

Results are written to `analysis_output_dir`. Each pipeline run also drops a timestamped
`pipeline_parallel_*.log` and a `config_*.yml` snapshot there. The main per-experiment output is:

- `{experiment}_PSM_RSM.csv` – PSMs annotated with their FDR-filtered RSM identification, the final result of the pipeline.

Intermediate per-experiment tables are written alongside it (RSM = RNA–spectrum match):

| File | What it holds |
| --- | --- |
| `{experiment}_low_id_ms2_df.csv` | Processed low-ID MS2 scans – the spectra fed into the RSM search |
| `{experiment}_high_ce_psm.csv` | Processed high-CE peptide identifications (PSMs) – the other search input |
| `{experiment}_RSM_matching_scores.csv` | Every matching candidate per scan (all scored candidates) |
| `{experiment}_low_id_ms2_df_RSM_report.csv` | All RSMs per scan, with the FDR verdict and per-scan E-values (see below) |
| `{experiment}_PSM_RSM.csv` | PSMs annotated with their RSM identification |

Manifest-level summary reports (e.g. `manifest_RSM_FDR_report.csv`) are written as well.

### The RSM report

`{experiment}_low_id_ms2_df_RSM_report.csv` holds **every scored row** – targets and non-canonical
decoys alike – rather than only the FDR survivors. Each scan contributes one winning hypothesis,
target and decoy competing within a single concatenated library of equal size; winners are ranked by
decreasing `Hyperscore_XL` and the FDR among reported targets is estimated as `(D + 1) / T` at each
score threshold. On top of the scoring columns the report adds:

| Column | Meaning |
| --- | --- |
| `is_decoy` | The row's `ID` matched one of `FDR_control.noncanonical_decoy_id_prefixes` |
| `FDR_cutoff` | Lowest accepted `Hyperscore_XL` (constant per experiment) – an output of the q-value filter, not its input |
| `pass_FDR_threshold` | `True` for target rows with `q_value <= FDR_control.rsm_fdr_level`; `False` for decoys and non-passing targets |
| `n_candidates`, `best_Evalue_XL`, `Evalue_notes` | Per-scan E-value results, computed for FDR-surviving scans only (empty elsewhere) |

The per-threshold counts, FDR estimates and q-values are deliberately **not** columns of the report:
each is a function of `Hyperscore_XL` and `is_decoy`, which every row carries, so storing them would
only restate what those two columns already determine. The cutoff's own diagnostics (cumulative T and
D, `estimated_FDR`, `q_value`) go to the step's log file.

**To get the accepted RSMs, filter on `pass_FDR_threshold == True`.** Retaining the decoy rows is
deliberate: `Hyperscore_XL` plus `is_decoy` is exactly the input the q-value calculation needs, so
the whole thing can be reproduced – or re-thresholded at a different `FDR_control.rsm_fdr_level` –
from this file alone, without re-running the pipeline.

## Tests

```bash
pytest
```

Tests cover the core pure functions (JSON array round-trip, RSM file resolution), the fragment-matching algorithm, and the Hyperscore computation. They are unit tests – they need no MS data and run in about a second.

## Repository layout

- `salt/` – the importable package: pipeline steps, the runner (`runner.py`), the shared
  utility module (`utils.py`), and the package entry point (`__main__.py`)
- `salt/data/` – reference data shipped with the package (theoretical-spectra libraries and the
  adduct-annotation table), loaded at runtime via `importlib.resources`
- `prepare/` – scripts to generate/customize the theoretical-spectra libraries in `salt/data/`,
  chained by `build_libraries.py` (run manually when analyzing a different modified nucleotide or
  changing the RNA moiety/mass definitions – see
  [Regenerating the theoretical-spectra libraries](#regenerating-the-theoretical-spectra-libraries))
- `tests/` – pytest unit tests


## License

Apache License 2.0 – see [LICENSE](LICENSE) and [NOTICE](NOTICE).
