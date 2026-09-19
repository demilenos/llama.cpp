[CmdletBinding()]
param(
    [Parameter(Mandatory)] [int] $ServerPid,
    [int] $Port = 9931,
    [string] $Prompt = ('Explain the following benchmark note in one paragraph: ' + ('memory ' * 256)),
    [int] $NPredict = 128,
    [string] $Luid = '0x0001205d',
    [int] $SampleMs = 150,
    [string] $OutputDir = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path 'pp-fix')
)

$ErrorActionPreference = 'Stop'
function ConvertTo-DedicatedMemorySample {
    param($CounterSet, [string] $AdapterLuid, [double] $TimeMs, [string] $Phase)
    $matching = @($CounterSet.CounterSamples | Where-Object { $_.InstanceName -match [regex]::Escape($AdapterLuid) })
    if (-not $matching.Count) { return $null }
    $total = 0.0
    $instances = [System.Collections.Generic.List[object]]::new()
    foreach ($sample in $matching) {
        # Reject incomplete snapshots rather than undercounting an adapter.
        if ($null -eq $sample.CookedValue -or $null -eq $sample.Status -or $sample.Status -ne 0) { return $null }
        try { $value = [double]$sample.CookedValue } catch { return $null }
        if ([double]::IsNaN($value) -or [double]::IsInfinity($value) -or $value -lt 0) { return $null }
        $total += $value
        $instances.Add([pscustomobject]@{ Instance = $sample.InstanceName; DedicatedMiB = $value / 1MB })
    }
    if ([double]::IsInfinity($total)) { return $null }
    [pscustomobject]@{
        TimeMs = [math]::Round($TimeMs, 2); CounterTimestamp = $CounterSet.Timestamp
        Phase = $Phase; DedicatedMiB = $total / 1MB; Instances = $instances
    }
}

$proc = Get-Process -Id $ServerPid -ErrorAction Stop
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.OwningProcess -eq $ServerPid }
if (-not $listener) {
    throw "No listener found on port $Port."
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = Join-Path $OutputDir "server-memory-$stamp.json"
$body = @{ prompt = $Prompt; n_predict = $NPredict; ignore_eos = $true; cache_prompt = $false; stream = $false } | ConvertTo-Json -Depth 4
$uri = "http://127.0.0.1:$Port/completion"
$handler = [System.Net.Http.HttpClientHandler]::new()
$client = [System.Net.Http.HttpClient]::new($handler)
$client.Timeout = [TimeSpan]::FromMinutes(30)
$content = [System.Net.Http.StringContent]::new($body, [Text.Encoding]::UTF8, 'application/json')
$sw = [Diagnostics.Stopwatch]::StartNew()
$requestTask = $client.PostAsync($uri, $content)
$samples = [System.Collections.Generic.List[object]]::new()
$counterPath = '\GPU Adapter Memory(*)\Dedicated Usage'
$failedSamples = 0
try {
    while (-not $requestTask.IsCompleted) {
        $sample = $null
        try {
            $c = Get-Counter -Counter $counterPath -ErrorAction Stop
            $sample = ConvertTo-DedicatedMemorySample $c $Luid $sw.Elapsed.TotalMilliseconds 'request'
        } catch { }
        if ($null -ne $sample) { $samples.Add($sample) } else { $failedSamples++ }
        Start-Sleep -Milliseconds $SampleMs
    }
    $response = $requestTask.GetAwaiter().GetResult()
    [void]$response.EnsureSuccessStatusCode()
    $sample = $null
    try {
        $c = Get-Counter -Counter $counterPath -ErrorAction Stop
        $sample = ConvertTo-DedicatedMemorySample $c $Luid $sw.Elapsed.TotalMilliseconds 'complete'
    } catch { }
    if ($null -ne $sample) { $samples.Add($sample) } else { $failedSamples++ }
    $requestSamples = @($samples | Where-Object Phase -eq 'request').Count
    if ($samples.Count -lt 2 -or $requestSamples -lt 1) {
        throw "Insufficient valid dedicated-memory samples for LUID $Luid ($requestSamples during request, $($samples.Count) total); refusing to report memory headroom."
    }
    $peak = ($samples | Measure-Object DedicatedMiB -Maximum).Maximum
    $text = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    $sw.Stop()
    $result = [pscustomobject]@{
        ServerPid = $ServerPid; Port = $Port; Luid = $Luid; CounterPath = $counterPath; RequestedSampleMs = $SampleMs; CounterSamplingNote = 'Get-Counter latency can exceed RequestedSampleMs; timestamps are measured intervals, not guaranteed cadence.'
        NPredict = $NPredict; PromptChars = $Prompt.Length; DurationMs = [math]::Round($sw.Elapsed.TotalMilliseconds,2)
        PeakDedicatedMiB = [math]::Round($peak,2); RemainingMiBOf8098 = [math]::Round(8098 - $peak,2)
        ValidSamples = $samples.Count; RequestSamples = $requestSamples; FailedSamples = $failedSamples
        Samples = $samples; Response = ($text | ConvertFrom-Json)
    }
    $result | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $out -Encoding UTF8
    $result | Select-Object ServerPid,Port,DurationMs,PeakDedicatedMiB,RemainingMiBOf8098,@{n='Evidence';e={$out}}
}
finally {
    $content.Dispose(); $client.Dispose(); $handler.Dispose()
}
