# CSC8114 Group Project — Paper

## Directory Structure

```
paper/
├── csc8114 accessment1.tex   # Main LaTeX source
├── refs.bib                  # Bibliography entries
├── IEEEabrv.bib              # IEEE abbreviations (do not edit)
├── IEEEtran.bst              # IEEE bibliography style (do not edit)
└── diagrams/
    ├── Architecture.mmd      # Mermaid source for architecture diagram
    ├── Architecture.png      # Generated PNG (committed to repo)
    ├── Sequence.mmd          # Mermaid source for sequence diagram
    └── Sequence.png          # Generated PNG (committed to repo)
```

---

## Generating the PDF

Run the following commands **in the `paper/` directory** in order:

```bash
cd paper/

pdflatex "csc8114 accessment1.tex"
bibtex "csc8114 accessment1"
pdflatex "csc8114 accessment1.tex"
pdflatex "csc8114 accessment1.tex"
```

> You need to run `pdflatex` **three times** to resolve all cross-references and citations correctly. The output will be `csc8114 accessment1.pdf`.

**Requirements:** [TeX Live](https://www.tug.org/texlive/) (recommended) or MiKTeX.

---

## Editing the Paper

The paper uses the **IEEE conference format** (`IEEEtran`). Each teammate edits their own section.

- Add a new section: `\section{Your Section Title}`
- Add a subsection: `\subsection{...}`
- Cite a reference: `\cite{key}` — the key must exist in `refs.bib`
- Reference a figure: `Fig.~\ref{fig:label}`

### Adding References

Add entries to `refs.bib` using BibTeX format, then cite with `\cite{key}`:

```bibtex
@article{smith2023,
  author  = {John Smith},
  title   = {Paper Title},
  journal = {Journal Name},
  year    = {2023},
  volume  = {1},
  pages   = {1--10}
}
```

Then in the `.tex` file: `...as shown by Smith \cite{smith2023}...`

---

## Editing Diagrams

Diagrams are written in [Mermaid](https://mermaid.js.org/) (`.mmd` files). **Always edit and export using [mermaid.live](https://mermaid.live/)** — the online renderer and the local `mmdc` CLI produce slightly different output, so we use the website as the single source of truth.

### Workflow

1. Open [https://mermaid.live](https://mermaid.live/)
2. Paste the contents of the relevant `.mmd` file (e.g. `diagrams/Architecture.mmd`)
3. Make your edits in the left panel — the preview updates in real time
4. When done, copy the updated Mermaid source back into the `.mmd` file
5. Download the PNG from the website (**Actions → Download PNG**, select background: white)
6. Save it as the corresponding file in `diagrams/` (e.g. `diagrams/Architecture.png`)
7. Commit both the updated `.mmd` and the new `.png`

> **Do not use `mmdc` locally** — the rendering differs from the website and will produce inconsistent diagrams.

### Current diagrams

| File                        | Description                          | Edit link                                     |
| --------------------------- | ------------------------------------ | --------------------------------------------- |
| `diagrams/Architecture.mmd` | System architecture (flowchart)      | [Open in mermaid.live](https://mermaid.live/) |
| `diagrams/Sequence.mmd`     | Training workflow (sequence diagram) | [Open in mermaid.live](https://mermaid.live/) |