# BioNeural — one-click GitHub push
# Creates the repo on GitHub (via the gh CLI) and pushes the entire project.
#
#   .\push.ps1                 # default: repo name = folder name, private
#   .\push.ps1 my-neural private
#   .\push.ps1 bioneural public
#
# Requires: gh (GitHub CLI) installed and authenticated: `gh auth login`
param(
    [string]$RepoName = "",
    [string]$Visibility = "private"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: GitHub CLI (gh) not found. Install from https://cli.github.com/ and run: gh auth login" -ForegroundColor Red
    exit 1
}

# --- determine repo name (default = current folder name) ---
if ([string]::IsNullOrWhiteSpace($RepoName)) {
    $RepoName = (Split-Path -Leaf (Get-Location))
}
$Visibility = $Visibility.ToLower()
if ($Visibility -notin @("private", "public")) {
    Write-Host "ERROR: visibility must be 'private' or 'public'" -ForegroundColor Red
    exit 1
}

Write-Host "==> Checking git identity..."
if (-not (git config user.name) -or -not (git config user.email)) {
    Write-Host "ERROR: git user.name / user.email not set. Run:" -ForegroundColor Red
    Write-Host "    git config --global user.name  \"Your Name\""
    Write-Host "    git config --global user.email \"you@example.com\""
    exit 1
}

Write-Host "==> Ensuring repo is initialized..."
if (-not (Test-Path ".git")) {
    git init -b main
}

Write-Host "==> Ensuring tokenizer/runtime envs don't leak..."
if (Test-Path ".env") {
    Write-Host "WARNING: .env exists — it is gitignored and will NOT be pushed."
}

Write-Host "==> Staging everything..."
git add -A
if (-not (git diff --cached --quiet)) {
    git commit -m "BioNeural v0.1 — event-driven ternary neural organism + benchmark harness" --allow-empty
} else {
    Write-Host "Nothing new to commit (already committed)."
}

Write-Host "==> Creating GitHub repo '$RepoName' ($Visibility)..."
$remote = gh repo view "$RepoName" --json nameWithOwner -q .nameWithOwner 2>$null
if (-not $remote) {
    gh repo create $RepoName --$Visibility --source . --remote origin --push
} else {
    if (-not (git remote | Select-String "^origin$")) {
        git remote add origin "https://github.com/$remote.git"
    }
    git push -u origin main
}

Write-Host ""
Write-Host "SUCCESS. Repo: https://github.com/$((gh repo view --json nameWithOwner -q .nameWithOwner))" -ForegroundColor Green