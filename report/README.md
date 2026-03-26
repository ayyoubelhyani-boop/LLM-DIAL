# Projet LaTeX du rapport

Ce dossier contient un squelette de rapport LaTeX pour le projet Dial-In LLM.

## Structure

- `main.tex` : point d entree du rapport
- `chapters/` : chapitres separes
- `references.bib` : bibliographie BibTeX
- `build.ps1` : script PowerShell de compilation

## Compilation

Compilation recommandee avec `latexmk` :

```powershell
pwsh -File .\build.ps1
```

Le script utilise `latexmk` si disponible, puis retombe sur `pdflatex` + `bibtex`.

Les artefacts de compilation sont places dans `build/`.

## Chapitres actuels

- `01_introduction.tex`
- `02_reproduction_papier.tex`
- `03_perspectives.tex`
- `04_synthese_comparative.tex`
- `05_conclusion.tex`
