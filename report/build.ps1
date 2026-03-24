$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$buildDir = Join-Path $root "build"
$mainTex = Join-Path $root "main.tex"

if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

$latexmk = Get-Command latexmk -ErrorAction SilentlyContinue
if ($latexmk) {
    & $latexmk.Source "-pdf" "-interaction=nonstopmode" "-synctex=1" "-outdir=build" $mainTex
    exit $LASTEXITCODE
}

$pdflatex = Get-Command pdflatex -ErrorAction SilentlyContinue
if (-not $pdflatex) {
    throw "Ni latexmk ni pdflatex ne sont disponibles dans le PATH."
}

Push-Location $root
try {
    & $pdflatex.Source "-interaction=nonstopmode" "-output-directory=build" "main.tex"
    if (Test-Path (Join-Path $buildDir "main.aux")) {
        $bibtex = Get-Command bibtex -ErrorAction SilentlyContinue
        if ($bibtex) {
            Push-Location $buildDir
            try {
                & $bibtex.Source "main"
            } finally {
                Pop-Location
            }
        }
    }
    & $pdflatex.Source "-interaction=nonstopmode" "-output-directory=build" "main.tex"
    & $pdflatex.Source "-interaction=nonstopmode" "-output-directory=build" "main.tex"
} finally {
    Pop-Location
}
