# RELAB Field Dictionary and Provenance Guide

**Date:** 2026-09-21
**Scope:** All metadata fields encountered in `data/raw/RELAB/` `.tab` files
**Sources:** Direct inspection of 1017 files and their trailing metadata blocks

---

## 1. File structure fields

| Field | Meaning | Source | Confidence | Missing possible? |
| ----- | ------- | ------ | ---------- | ----------------- |
| Filename | Encodes instrument prefix + sample code + optional suffix | Filename | High | No |
| Mineral directory | Mineral class label (unverified) | Directory name | Medium | No |
| Line 1 (header) | Declared number of spectral data rows | File line 1 | High | No |
| Column 1 | Wavelength in nanometers (nm) | Data block | High | No |
| Column 2 | Reflectance (unitless, 0-1 range) | Data block | High | No |
| Column 3 | Measurement uncertainty (unitless) — present in 461/1017 files | Data block | High | Yes |

---

## 2. Trailing metadata fields (present in all files)

| Field | Meaning | Source | Confidence | Missing possible? |
| ----- | ------- | ------ | ---------- | ----------------- |
| .ASC identifier | Original RELAB file identifier (uppercase filename + `.ASC`) | Metadata line 1 | High | No |
| Sample ID | RELAB sample identifier (format: `XX-YYY-ZZZ[-SUFFIX]`) | Metadata line 2 | High | No |
| Sample description | Mineral name, grain size, preparation notes | Metadata line 3+ | High | No |
| Splice information | "Connected at X um" — spectral range splice points | Metadata | High | Yes (286/1017) |
| Source angle | Illumination geometry (degrees) | `Source Ang:` | High | Yes (39/1017) |
| Detector angle | Observation geometry (degrees) | `Detect Ang:` | High | Yes (39/1017) |
| Detector voltage | Instrument voltage setting | `Volt:` | High | Yes (39/1017) |
| Measurement date | Date of measurement | `Date:` | High | No |
| Measurement time | Time of measurement | `Time:` | High | No |

---

## 3. Extended metadata fields (present in 341/1017 files)

These fields appear only in files with the extended structured metadata block (lines prefixed with `*`):

| Field | Meaning | Source | Confidence | Missing possible? |
| ----- | ------- | ------ | ---------- | ----------------- |
| User | Operator initials | `* User:` | High | Yes (676/1017) |
| Research type | Program/campaign type | `* Research Type:` | High | Yes (676/1017) |
| NASA PI/CoI or Sponsor | Principal investigator or sponsor code | `* NASA PI/CoI or Sponsor:` | High | Yes (676/1017) |
| BVC | Brown University affiliation | `* BVC:` | High | Yes (676/1017) |
| Public release date | Date when data was made public | `* Public Release Date:` | High | Yes (676/1017) |
| Run mode | Measurement mode | `* Run Mode:` | High | Yes (680/1017) |
| Stage type | Sample stage type | `* Stage Type:` | High | Yes (680/1017) |
| Spinning | Whether sample was spinning during measurement | `* Spinning:` | High | Yes (680/1017) |
| Aperture size | Instrument aperture size (mm) | `* Aperture Size:` | High | Yes (676/1017) |
| Scrambler | Scrambler/depolarizer state | `* Scrambler:` | High | Yes (676/1017) |
| Depolarizer | Depolarizer state | `* Depolarizer:` | High | Yes (676/1017) |
| Azimuth angle | Azimuth angle (degrees) | `* Azimuth Angle:` | High | Yes (most files) |
| Initial background | Background measurement at home position | `* Initial Background at Home:` | High | Yes (most files) |
| Halon calibration note | "RATIOS ADJUSTED USING NBS HALON CALIBRATION TABLE" | Metadata | High | Yes (471/1017) |

---

## 4. Derived fields

| Field | Meaning | Source | Confidence | Missing possible? |
| ----- | ------- | ------ | ---------- | ----------------- |
| Instrument prefix | Spectrometer/configuration code | Filename prefix | Medium | No |
| Sample code | Compressed sample ID (first two components) | Filename | High | No |
| Suffix | Variant indicator (grain size, preparation, or cf) | Filename suffix | High | Yes |
| Measurement geometry | Combination of source and detector angles | Metadata | High | Yes |
| Spectral range | Min and max wavelength | Data block | High | No |
| Number of points | Number of spectral data points | Data block | High | No |

---

## 5. Fields that are ABSENT

The following information is **not present** in any raw file:

| Missing field | Impact |
| ------------- | ------ |
| Chemical composition (formula, wt% oxides, trace elements) | Cannot correlate chemistry with spectral features |
| Crystal structure (space group, unit cell parameters) | Cannot directly test crystal-structure hypotheses |
| Verified mineral identification method | Unknown — directory name only |
| Sample locality / geographic provenance | No geographic or source-locality information |
| Numeric grain size (in most files) | Only qualitative notes exist |
| Standardized sample preparation protocol | Ad hoc notes only |
| Measurement temperature / pressure / atmosphere | Not recorded |
| Instrument model or serial number | Not recorded — only implicit from prefix codes |
| Calibration certificate details | Not recorded |
| Unique specimen identifier | No persistent ID across the dataset |
| Replicate measurement grouping | Cannot determine which files are repeats of the same specimen |
| Phase purity or impurity data | Not recorded |
| Bibliographic references | Not recorded |
| Optical class (e.g., birefringent, opaque) | Not recorded |

---

## 6. Sample ID structure

The RELAB sample ID follows the format:

```
XX-YYY-ZZZ[-SUFFIX]
```

| Component | Meaning | Example |
| --------- | ------- | ------- |
| `XX` | 2-letter institution/sample collection code | `JB`, `PA`, `AG`, `MR`, `CZ` |
| `YYY` | 3-letter researcher/program code | `JLB`, `CMP`, `TJM`, `MSR`, `C1M` |
| `ZZZ` | 3-digit sample number | `549`, `005`, `009`, `004` |
| `SUFFIX` | Variant indicator (optional) | `-A`, `-B`, `-CF`, `-P`, `-C2` |

### Suffix meanings

| Suffix | Meaning |
| ------ | ------- |
| (none) | Standard measurement |
| `-A`, `-B`, `-C`, etc. | Grain-size fraction or preparation variant |
| `-CF` | Container filter blank measurement |
| `-P` | Different preparation / grain size |
| `-C2` | Variant (meaning unclear) |

---

## 7. Instrument prefix codes

| Prefix | Interpretation | Voltage | Geometry | Date range |
| ------ | -------------- | ------- | -------- | ---------- |
| `bir1`, `bir2`, `bmr1` | RELAB bidirectional spectrometer | 1.222 | (30, 0) | 2002-2011 |
| `c1` | RELAB spectrometer, 2011-2012 campaign | 0.777 | (30, 0) | 2011-2012 |
| `c2`, `c3`, `c5` | RELAB spectrometer, early era | 0.222-0.777 | (30, 0) | 1987-1998 |
| `ca`-`cg` | Grain-size separates of same sample | 0.222-0.777 | (30, 0) | 1992-2012 |
| `nn`, `n`, `m` | Off-axis measurement | 1.222 | (30, 30) | 1992-1997 |
| `1101` | Numeric campaign, off-axis | 1.222 | (30, 30) | 2001-2002 |
| `397` | V. Hamilton campaign WAR-6454 | 1.222 | (30, 30) | 1997 |

**Note:** These interpretations are based on observed patterns, not official documentation.

---

## 8. Caveats

- **No official RELAB documentation** was available online for this study.
- The RELAB website mentions a Users Manual available by contacting Ralph Milliken or Takahiro Hiroi, but this manual was not accessible.
- All interpretations are based on direct inspection of the data files and logical inference.
- Some field meanings may be incorrect or incomplete without official documentation.
