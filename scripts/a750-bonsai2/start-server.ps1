[CmdletBinding()]
param(
    [string] $WorkspaceRoot,
    [int] $ContextSize = 65535,
    [int] $Port = 9931,
    [string] $HostAddress = '127.0.0.1',
    [switch] $CpuEmbedding = $true,
    [int] $MmvWorkgroup = 64,
    [int] $MmvRows = 4,
    [int] $FwhtWorkgroup = 256,
    [int] $SubmitDivisor = 16,
    [int] $MaxNodesPerSubmit = 256,
    [switch] $HybridFwht = $true,
    [switch] $FuseSigns = $true
)

$ErrorActionPreference = 'Stop'
$root = if ($WorkspaceRoot) { (Resolve-Path $WorkspaceRoot).Path } else { (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path }
$model = Join-Path $root 'Ternary-Bonsai-2-27B-PTQ1_0.gguf'
$server = Join-Path $root 'prismml-llama.cpp\build-a750-vulkan\bin\llama-server.exe'
if (-not (Test-Path -LiteralPath $server)) { throw "A750 llama-server.exe not found: $server" }
if (-not (Test-Path -LiteralPath $model)) { throw "Original GGUF not found: $model" }

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is already listening; refusing to kill or reuse the existing process."
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logDir = Join-Path $root 'pp-fix'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir "a750-server-$stamp.stdout.log"
$stderr = Join-Path $logDir "a750-server-$stamp.stderr.log"
$old = @{}
foreach ($name in 'GGML_VK_DISABLE_MMVQ','GGML_VK_A750_FWHT_SHMEM','GGML_VK_A750_FWHT_WG','GGML_VK_A750_FWHT_HYBRID','GGML_VK_A750_FWHT_SIGNS','GGML_VK_PTQ1_MMV_WG','GGML_VK_PTQ1_MMV_ROWS','GGML_VK_PTQ1_FORCE_XMX','GGML_VK_SERIALIZE_SUBMISSIONS','GGML_VK_A750_SUBMIT_DIVISOR','GGML_VK_MAX_NODES_PER_SUBMIT','GGML_VK_PTQ1_TILE','GGML_VK_PERF_LOGGER','GGML_VK_PTQ1_PROBE','GGML_VK_DISABLE_FUSION','GGML_VK_SUBMIT_TRACE') {
    $old[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
try {
    $env:GGML_VK_DISABLE_MMVQ = '1'
    $env:GGML_VK_A750_FWHT_SHMEM = '1'
    $env:GGML_VK_PTQ1_MMV_WG = "$MmvWorkgroup"
    $env:GGML_VK_PTQ1_MMV_ROWS = "$MmvRows"
    $env:GGML_VK_A750_FWHT_WG = "$FwhtWorkgroup"
    if ($HybridFwht) { $env:GGML_VK_A750_FWHT_HYBRID = '1' } else { Remove-Item Env:GGML_VK_A750_FWHT_HYBRID -ErrorAction SilentlyContinue }
    if ($FuseSigns) { $env:GGML_VK_A750_FWHT_SIGNS = '1' } else { Remove-Item Env:GGML_VK_A750_FWHT_SIGNS -ErrorAction SilentlyContinue }
    $env:GGML_VK_A750_SUBMIT_DIVISOR = "$SubmitDivisor"
    $env:GGML_VK_MAX_NODES_PER_SUBMIT = "$MaxNodesPerSubmit"
    Remove-Item Env:GGML_VK_PTQ1_FORCE_XMX,Env:GGML_VK_SERIALIZE_SUBMISSIONS,Env:GGML_VK_PTQ1_TILE,Env:GGML_VK_PERF_LOGGER,Env:GGML_VK_PTQ1_PROBE,Env:GGML_VK_DISABLE_FUSION,Env:GGML_VK_SUBMIT_TRACE -ErrorAction SilentlyContinue

    $embeddingOverride = if ($CpuEmbedding) { 'token_embd.weight=CPU' } else { 'token_embd.weight=Vulkan1' }
    $serverArgs = @('-m', ('"' + $model + '"'), '--host', $HostAddress, '--port', "$Port", '-c', "$ContextSize", '-ngl', '99', '--device', 'Vulkan1', '-ot', $embeddingOverride, '-ctk', 'q8_0', '-ctv', 'q8_0', '-np', '1', '-b', '512', '-ub', '512', '-fa', 'on', '--jinja', '--temp', '0.7', '--top-p', '0.95', '--top-k', '20')
    $proc = Start-Process -FilePath $server -ArgumentList $serverArgs -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    [pscustomobject]@{ PID = $proc.Id; ContextSize = $ContextSize; HostAddress = $HostAddress; Port = $Port; Server = $server; StdoutLog = $stdout; StderrLog = $stderr; CpuEmbedding = [bool]$CpuEmbedding; MmvWorkgroup = $MmvWorkgroup; MmvRows = $MmvRows; FwhtWorkgroup = $FwhtWorkgroup; SubmitDivisor = $SubmitDivisor; MaxNodesPerSubmit = $MaxNodesPerSubmit; HybridFwht = [bool]$HybridFwht; FuseSigns = [bool]$FuseSigns; Environment = @{ GGML_VK_DISABLE_MMVQ='1'; GGML_VK_PTQ1_MMV_WG="$MmvWorkgroup"; GGML_VK_PTQ1_MMV_ROWS="$MmvRows"; GGML_VK_A750_FWHT_WG="$FwhtWorkgroup"; GGML_VK_A750_SUBMIT_DIVISOR="$SubmitDivisor"; GGML_VK_MAX_NODES_PER_SUBMIT="$MaxNodesPerSubmit" } }
}
finally {
    foreach ($name in $old.Keys) {
        if ($null -eq $old[$name]) { Remove-Item "Env:$name" -ErrorAction SilentlyContinue }
        else { Set-Item "Env:$name" $old[$name] }
    }
}
