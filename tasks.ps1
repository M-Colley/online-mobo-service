#!/usr/bin/env pwsh
# Common dev/deploy tasks for the VAM optimizer service.
#   .\tasks.ps1 <task>
# Tasks: describe | unit | test | inspect | deploy | help
param([Parameter(Position = 0)][string]$Task = "help")

$ErrorActionPreference = "Stop"
$Region  = $env:REGION  ; if (-not $Region)  { $Region  = "us-central1" }
$Project = $env:PROJECT   # required for deploy: $env:PROJECT = "my-gcp-project"
$Service = $env:SERVICE ; if (-not $Service) { $Service = "vam-optimizer" }

$modules = @("space.py", "optimizer_core.py", "main.py",
             "simulate.py", "tests/test_service.py", "tests/test_space.py", "inspect_db.py")

switch ($Task) {
    "describe" { python space.py }
    "unit"     { python tests/test_space.py }
    "test" {
        python -m py_compile @modules
        python tests/test_space.py
        python simulate.py
        python tests/test_service.py
    }
    "inspect" { python inspect_db.py }
    "deploy" {
        if (-not $Project) { throw "Set PROJECT first, e.g.:  `$env:PROJECT = 'my-gcp-project'; .\tasks.ps1 deploy" }
        gcloud run deploy $Service --source . --project $Project `
            --region $Region --allow-unauthenticated `
            --memory 2Gi --cpu 2 --timeout 300
    }
    default {
        Write-Host "Usage: .\tasks.ps1 <task>"
        Write-Host "  describe  print the resolved JND grids"
        Write-Host "  unit      search-space unit tests (~4 s, exhaustive rules check)"
        Write-Host "  test      compile + unit + simulate + full service test"
        Write-Host "  inspect   read-only Firestore inspector (needs auth + GOOGLE_CLOUD_PROJECT)"
        Write-Host "  deploy    gcloud run deploy (set `$env:PROJECT first; region: $Region)"
    }
}
