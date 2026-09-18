[CmdletBinding()]
param(
    [string]$BuildDir = "build-ternary-alchemist-v1",
    [string]$Model = "",
    [ValidateSet(0, 1, 2, 4)]
    [int]$Tile = 0,
    [ValidateSet(4, 8, 16, 32)]
    [int]$LocalSize = 8,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "../..")
$BaseBuild = Join-Path $PSScriptRoot "build_ptq1_xmx_v5.ps1"

$Args = @{
    BuildDir = $BuildDir
}
if ($Clean) { $Args.Clean = $true }

# Reuse the proven v5 configure/toolchain checks but keep a distinct build tree.
& $BaseBuild @Args
if ($LASTEXITCODE -ne 0) { throw "Ternary optimized build failed." }

if (-not $Model) {
    Write-Host "Ternary build/tests passed: $BuildDir"
    Write-Host "Use -Model <ptq1.gguf> to run the A750 smoke."
    return
}

$ModelPath = Resolve-Path $Model
$Bench = Join-Path $Repo "$BuildDir/bin/llama-bench.exe"
if (-not (Test-Path -LiteralPath $Bench -PathType Leaf)) {
    throw "llama-bench.exe not found: $Bench"
}

$env:GGML_VULKAN_PTQ1_XMX = "1"
if ($Tile -eq 0) {
    Remove-Item Env:GGML_VULKAN_PTQ1_XMX_TILE -ErrorAction SilentlyContinue
    Write-Host "PTQ1 token tile: auto (n=2 -> T2, n=4 -> T4, other n -> baseline)"
} else {
    $env:GGML_VULKAN_PTQ1_XMX_TILE = [string]$Tile
    Write-Host "PTQ1 token tile: forced T$Tile"
}
$env:GGML_VULKAN_PTQ1_XMX_LOCAL_SIZE = [string]$LocalSize
Write-Host "PTQ1 local size: $LocalSize"

& $Bench -m $ModelPath -ngl 99 -p 512 -n 128
if ($LASTEXITCODE -ne 0) { throw "Ternary A750 smoke failed." }
