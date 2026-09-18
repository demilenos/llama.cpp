[CmdletBinding()]
param(
    [string]$BuildDir = "build-ptq1-xmx",
    [string]$Model = "",
    [string]$CCompiler = "",
    [string]$CxxCompiler = "",
    [string]$VulkanSdk = $env:VULKAN_SDK,
    [string]$LevelZeroSdk = $env:LEVEL_ZERO_V1_SDK_PATH,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "../..")

if (-not $LevelZeroSdk) { $LevelZeroSdk = $env:LEVEL_ZERO_SDK_PATH }
if (-not $VulkanSdk) { throw "Set VULKAN_SDK or pass -VulkanSdk." }
if (-not $LevelZeroSdk) { throw "Set LEVEL_ZERO_V1_SDK_PATH/LEVEL_ZERO_SDK_PATH or pass -LevelZeroSdk." }

$Required = @(
    (Join-Path $VulkanSdk "Include/vulkan/vulkan.h"),
    (Join-Path $VulkanSdk "Lib/vulkan-1.lib"),
    (Join-Path $LevelZeroSdk "include/level_zero/ze_api.h"),
    (Join-Path $LevelZeroSdk "lib/ze_loader.lib")
)
foreach ($Path in $Required) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing SDK file: $Path"
    }
}

if (-not $CCompiler) {
    $Cmd = Get-Command icx.exe -ErrorAction SilentlyContinue
    if ($Cmd) { $CCompiler = $Cmd.Source }
}
if (-not $CxxCompiler) {
    foreach ($Name in @("icpx.exe", "icx.exe")) {
        $Cmd = Get-Command $Name -ErrorAction SilentlyContinue
        if ($Cmd) { $CxxCompiler = $Cmd.Source; break }
    }
}
if (-not $CCompiler -or -not $CxxCompiler) {
    throw "Initialize the oneAPI environment first; icx and icpx/icx must be on PATH."
}

$Ninja = Get-Command ninja.exe -ErrorAction SilentlyContinue
if (-not $Ninja) { throw "ninja.exe is required for this smoke build." }

$Build = Join-Path $Repo $BuildDir
if ($Clean -and (Test-Path $Build)) {
    Remove-Item -LiteralPath $Build -Recurse -Force
}

Push-Location $Repo
try {
    $CmakeArgs = @(
        "-S", ".",
        "-B", $Build,
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_C_COMPILER=$CCompiler",
        "-DCMAKE_CXX_COMPILER=$CxxCompiler",
        "-DGGML_VULKAN=ON",
        "-DLLAMA_PTQ1_XMX=ON",
        "-DLLAMA_BUILD_TESTS=ON",
        "-DLLAMA_BUILD_TOOLS=ON"
    )
    & cmake @CmakeArgs
    if ($LASTEXITCODE -ne 0) { throw "PTQ1 XMX CMake configure failed." }

    & cmake --build $Build --target maple-ptq1-repack-tests test-quantize-fns llama-bench -j
    if ($LASTEXITCODE -ne 0) { throw "PTQ1 XMX build failed." }

    & ctest --test-dir $Build --output-on-failure -R "maple-ptq1-repack-tests|test-quantize-fns"
    if ($LASTEXITCODE -ne 0) { throw "PTQ1 host regression tests failed." }

    if ($Model) {
        $ModelPath = Resolve-Path $Model
        $env:GGML_VULKAN_PTQ1_XMX = "1"
        $Bench = Join-Path $Build "bin/llama-bench.exe"
        if (-not (Test-Path -LiteralPath $Bench -PathType Leaf)) {
            throw "llama-bench.exe not found: $Bench"
        }
        Write-Host "Running PTQ1 XMX smoke. Look for: PTQ1 XMX executor registered / ptq1-xmx ..."
        & $Bench -m $ModelPath -ngl 99 -p 512 -n 128
        if ($LASTEXITCODE -ne 0) { throw "PTQ1 XMX llama-bench smoke failed." }
    } else {
        Write-Host "Build/tests passed. Pass -Model <ptq1.gguf> to run the A750 smoke test."
    }
} finally {
    Pop-Location
}
