$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$settings = @{
    GGML_VK_PTQ1_MMV_WG = '64'
    GGML_VK_PTQ1_MMV_ROWS = '4'
    GGML_VK_DISABLE_MMVQ = '1'
    GGML_VK_A750_FWHT_SHMEM = '1'
    GGML_VK_A750_FWHT_WG = '256'
    GGML_VK_A750_FWHT_HYBRID = '1'
    GGML_VK_A750_FWHT_SIGNS = '1'
    GGML_VK_A750_SUBMIT_DIVISOR = '16'
    GGML_VK_MAX_NODES_PER_SUBMIT = '256'
    GGML_VK_PTQ1_FORCE_XMX = $null
    GGML_VK_SERIALIZE_SUBMISSIONS = $null
    GGML_VK_PTQ1_TILE = $null
    GGML_VK_PERF_LOGGER = $null
    GGML_VK_PTQ1_PROBE = $null
    GGML_VK_DISABLE_FUSION = $null
    GGML_VK_SUBMIT_TRACE = $null
}
$previous = @{}
foreach ($key in $settings.Keys) {
    $previous[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
    if ($null -eq $settings[$key]) { [Environment]::SetEnvironmentVariable($key, $null, 'Process') }
    else { [Environment]::SetEnvironmentVariable($key, $settings[$key], 'Process') }
}
try {
    & (Join-Path $workspaceRoot 'prismml-llama.cpp\build-a750-vulkan\bin\llama-perplexity.exe') `
        -m (Join-Path $workspaceRoot 'Ternary-Bonsai-2-27B-PTQ1_0.gguf') `
        --device Vulkan1 -ngl 99 -ot 'token_embd.weight=CPU' `
        -f (Join-Path $workspaceRoot 'pp-fix\wikitext-2-raw\wiki.test.raw') `
        -c 512 -b 512 -ub 1 --chunks 4 `
        --kl-divergence-base (Join-Path $workspaceRoot 'pp-fix\baseline.kld') `
        --kl-divergence -ctk q8_0 -ctv q8_0 -fa on
    if ($LASTEXITCODE -ne 0) { throw "llama-perplexity failed: $LASTEXITCODE" }
} finally {
    foreach ($key in $settings.Keys) {
        [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process')
    }
}
