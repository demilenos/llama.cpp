param(
    [string] $OutputRoot = "C:\AI\llama-maple-integration\bench-vulkan1-40k-20260918"
)

$ErrorActionPreference = "Stop"

function Invoke-MapleBenchmark {
    param([string] $OutDir)

    $model = "C:\AI\models\maple-preview-TQ2_0-head-F16.gguf"
    $corpus = "C:\Users\demil\Documents\Codex\2026-09-17\spark2-5-n1-models-spark2-5\corpora\test.corpus.40k"
    $setvars = "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
    $upstream = "C:\AI\llama-upstream-vulkan\build\bin\llama.exe"
    $custom = "C:\AI\llama-maple-integration\build-maple\bin\llama-completion.exe"

    foreach ($path in @($model, $corpus, $setvars, $upstream, $custom)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Required benchmark input is missing: $path"
        }
    }

    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $cases = @(
        [pscustomobject] @{ Name = "upstream_default"; Exe = $upstream; Level4 = $null; Tile = $null },
        [pscustomobject] @{ Name = "custom_l0_t1"; Exe = $custom; Level4 = "0"; Tile = "1" },
        [pscustomobject] @{ Name = "custom_l0_t4"; Exe = $custom; Level4 = "0"; Tile = "4" },
        [pscustomobject] @{ Name = "custom_l0_t8"; Exe = $custom; Level4 = "0"; Tile = "8" },
        [pscustomobject] @{ Name = "custom_l1_t1"; Exe = $custom; Level4 = "1"; Tile = "1" },
        [pscustomobject] @{ Name = "custom_l1_t4"; Exe = $custom; Level4 = "1"; Tile = "4" },
        [pscustomobject] @{ Name = "custom_l1_t8"; Exe = $custom; Level4 = "1"; Tile = "8" }
    )

    $commonArgs = @(
        "-m", $model, "-f", $corpus, "-ub", "512", "-b", "512", "-c", "49152",
        "-n", "4096", "--ignore-eos", "--no-conversation", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
        "-ngl", "999", "--device", "Vulkan1"
    )
    $results = [System.Collections.Generic.List[object]]::new()

    foreach ($case in $cases) {
        $caseDir = Join-Path $OutDir $case.Name
        New-Item -ItemType Directory -Force -Path $caseDir | Out-Null
        $stdout = Join-Path $caseDir "stdout.log"
        $stderr = Join-Path $caseDir "stderr.log"
        $wrapper = Join-Path $caseDir "run.cmd"
        $commandRecord = Join-Path $caseDir "command.txt"
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $case.Exe).Hash.ToLowerInvariant()
        $artifactHashes = [ordered]@{}
        foreach ($artifact in @($case.Exe, (Join-Path (Split-Path $case.Exe) 'llama.dll'), (Join-Path (Split-Path $case.Exe) 'ggml-vulkan.dll'), (Join-Path (Split-Path $case.Exe) 'llama-completion-impl.dll'))) {
            $artifactHashes[$artifact] = if (Test-Path -LiteralPath $artifact) { (Get-FileHash -Algorithm SHA256 -LiteralPath $artifact).Hash } else { $null }
        }
        $subcommand = if ($case.Name -eq 'upstream_default') { 'completion ' } else { '' }
        $argText = ($commonArgs | ForEach-Object { '"' + $_.Replace('"', '""') + '"' }) -join ' '
        $level4Set = if ($null -eq $case.Level4) { 'set "LLAMA_MAPLE_LEVEL4="' } else { 'set "LLAMA_MAPLE_LEVEL4=' + $case.Level4 + '"' }
        $tileSet = if ($null -eq $case.Tile) { 'set "LLAMA_MAPLE_TOKEN_TILE="' } else { 'set "LLAMA_MAPLE_TOKEN_TILE=' + $case.Tile + '"' }
        $wrapperText = @(
            '@echo off'
            'setlocal'
            ('call "' + $setvars + '" intel64 vs2022 >nul 2>&1')
            'set "GGML_BACKEND_PATH="'
            $level4Set
            $tileSet
            ('"' + $case.Exe + '" ' + $subcommand + $argText + ' 1>"' + $stdout + '" 2>"' + $stderr + '"')
            'exit /b %ERRORLEVEL%'
        ) -join [Environment]::NewLine
        Set-Content -LiteralPath $wrapper -Value $wrapperText -Encoding ascii

        $recorded = @(
            "case=$($case.Name)"
            "executable=$($case.Exe)"
            "sha256=$hash"
            "model=$model"
            "corpus=$corpus"
            "device=Vulkan1"
            "args=$subcommand$argText"
            "env=GGML_BACKEND_PATH=<unset>; LLAMA_MAPLE_LEVEL4=$($case.Level4); LLAMA_MAPLE_TOKEN_TILE=$($case.Tile)"
            "wrapper=$wrapper"
        ) -join [Environment]::NewLine
        Set-Content -LiteralPath $commandRecord -Value $recorded -Encoding utf8NoBOM

        $startedUtc = [DateTime]::UtcNow.ToString("o")
        $timer = [Diagnostics.Stopwatch]::StartNew()
        $proc = Start-Process -FilePath $env:ComSpec -ArgumentList @('/d', '/s', '/c', ('"' + $wrapper + '"')) -WorkingDirectory $caseDir -WindowStyle Hidden -Wait -PassThru
        $timer.Stop()
        $endedUtc = [DateTime]::UtcNow.ToString("o")
        $errorText = if (Test-Path -LiteralPath $stderr) { Get-Content -LiteralPath $stderr -Raw -ErrorAction SilentlyContinue } else { '' }
        $errorHit = [regex]::IsMatch($errorText, '(?im)\b(error|fatal|failed|failure|exception|abort(?:ed)?)\b')
        $status = if ($proc.ExitCode -ne 0) { 'exit_failure' } elseif ($errorHit) { 'stderr_error_exit0' } else { 'ok' }
        $results.Add([pscustomobject]@{
            case = $case.Name; executable = $case.Exe; sha256 = $hash; level4 = $case.Level4; token_tile = $case.Tile
            artifact_hashes = $artifactHashes; started_utc = $startedUtc; ended_utc = $endedUtc
            exit_code = $proc.ExitCode; wall_ms = $timer.ElapsedMilliseconds; status = $status
            stderr_error_pattern = $errorHit; stdout = $stdout; stderr = $stderr; command = $commandRecord
        })
        $results | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutDir "summary.json") -Encoding utf8NoBOM
        $results | Format-Table -AutoSize | Out-File -LiteralPath (Join-Path $OutDir "summary.txt") -Encoding utf8
    }

    $results | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutDir "summary.json") -Encoding utf8NoBOM
    $results | Format-Table -AutoSize | Out-File -LiteralPath (Join-Path $OutDir "summary.txt") -Encoding utf8
}

# Dot-sourcing defines the function without starting any GPU work.
if ($MyInvocation.InvocationName -ne '.') {
    Invoke-MapleBenchmark -OutDir $OutputRoot
}
