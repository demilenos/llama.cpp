[CmdletBinding()]
param(
    [string]$Compiler='', [string]$Device='A750', [int]$Tokens=1, [int]$Repeats=6,
    [string]$VulkanSdk=$env:VULKAN_SDK,
    [string]$LevelZeroSdk=$env:LEVEL_ZERO_V1_SDK_PATH,
    [switch]$MapleShape, [switch]$DenseA8, [switch]$Validation, [switch]$BuildOnly,
    [ValidateSet('advance','bootstrap','terminal')][string]$Entry='advance',
    [ValidateSet('direct','grouped','auto')][string]$Schedule='direct'
)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
if(-not $LevelZeroSdk){$LevelZeroSdk=$env:LEVEL_ZERO_SDK_PATH}
if(-not $VulkanSdk){throw 'Set VULKAN_SDK or pass -VulkanSdk (native Vulkan headers and vulkan-1.lib required).'}
if(-not $LevelZeroSdk){throw 'Set LEVEL_ZERO_V1_SDK_PATH or pass -LevelZeroSdk.'}
$VkInclude=Join-Path $VulkanSdk 'Include'
$VkLib=Join-Path $VulkanSdk 'Lib/vulkan-1.lib'
$ZeInclude=Join-Path $LevelZeroSdk 'include'
$ZeLib=Join-Path $LevelZeroSdk 'lib/ze_loader.lib'
foreach($P in @((Join-Path $VkInclude 'vulkan/vulkan.h'),$VkLib,(Join-Path $ZeInclude 'level_zero/ze_api.h'),$ZeLib)){
    if(-not (Test-Path -LiteralPath $P -PathType Leaf)){throw "Missing native SDK file: $P"}
}
if($Tokens -lt 1 -or $Repeats -lt 1 -or $Repeats -gt 1000){throw 'Invalid Tokens/Repeats'}
# Build the user's exact merged module. Do NOT replace old source / reuse foreign .obj files.
& (Join-Path $PSScriptRoot 'build_level4_and_test.ps1') -Compiler $Compiler -BuildOnly
if(-not $Compiler){foreach($Name in @('icpx.exe','icx.exe')){$C=Get-Command $Name -ErrorAction SilentlyContinue;if($C){$Compiler=$C.Source;break}}}
if(-not $Compiler){throw 'Initialize MSVC + oneAPI DPC++ first.'}
$F=@('-fexceptions','-fcxx-exceptions','-std=c++17','-fsycl','-O3','-fno-fast-math',
     '-fsycl-device-code-split=per_kernel','-Iinclude','-Iintegration',"-I$VkInclude","-I$ZeInclude")
$Extra=@();$Resource=& $Compiler '-print-resource-dir'
if($LASTEXITCODE -eq 0 -and $Resource){$P=Join-Path (([string](@($Resource)[-1])).Trim()) 'lib/windows/clang_rt.builtins-x86_64.lib';if(Test-Path $P){$Extra+=@($P)}}
function RunLogged([string[]]$ArgsList,[string]$Log){
    $Saved=$ErrorActionPreference
    try{$ErrorActionPreference='Continue';& $Compiler @ArgsList 2>&1 | Tee-Object -FilePath $Log;$Code=$LASTEXITCODE}
    finally{$ErrorActionPreference=$Saved}
    if($Code -ne 0){throw "Zero-copy build failed ($Code); inspect $Log. No server DLL changed."}
}
Push-Location $Root
try {
    RunLogged ($F+@('-c','integration/level4_zerocopy_win32.cpp','-o','build/level4_zerocopy_win32.obj')) 'build/level4-zero-copy-compile.log'
    $Objects=@('maple_w2a16','maple_w2a8','maple_moe','expert_grouping','maple_w2a8_grouped','maple_level4','level4_zerocopy_win32')|ForEach-Object{"build/$_.obj"}
    RunLogged ($F+@('tools/level4_zerocopy_probe.cpp')+$Objects+@($VkLib,$ZeLib)+$Extra+@('-o','build/maple-level4-zerocopy-probe.exe')) 'build/level4-zero-copy-link.log'
    & python tools/stamp_level4_build.py --root . --exe build/maple-level4-zerocopy-probe.exe
    if($LASTEXITCODE -ne 0){throw 'Build stamp failed'}
    Copy-Item -LiteralPath build/level4-build.json -Destination build/level4-zero-copy-build.json -Force
    if(-not $BuildOnly){
        $Out='build/level4-zero-copy-'+(Get-Date -Format 'yyyyMMdd-HHmmss-fff')
        New-Item -ItemType Directory -Path $Out -ErrorAction Stop | Out-Null
        Copy-Item build/level4-zero-copy-build.json $Out
        $Arguments=@('--device',$Device,'--tokens',"$Tokens",'--repeats',"$Repeats",'--entry',$Entry,'--schedule',$Schedule,'--out',$Out)
        if($MapleShape){$Arguments+=@('--maple')};if($DenseA8){$Arguments+=@('--dense-a8')};if($Validation){$Arguments+=@('--validation')}
        $Saved=$ErrorActionPreference
        try{$ErrorActionPreference='Continue';& build/maple-level4-zerocopy-probe.exe @Arguments 2>&1 | Tee-Object -FilePath (Join-Path $Out 'probe.log');$Code=$LASTEXITCODE}
        finally{$ErrorActionPreference=$Saved}
        if($Code -ne 0){throw "Native zero-copy test failed ($Code); preserve $Out; no copy fallback was attempted."}
        if(Select-String -Path (Join-Path $Out 'probe.log') -Pattern 'Validation Error|VUID-' -Quiet){throw "Vulkan validation messages require review: $Out"}
        & python tools/analyze_level4_zero_copy.py $Out
        if($LASTEXITCODE -ne 0){throw "Zero-copy result audit failed: $Out"}
        Write-Host "Results: $Out (synthetic native interop test, NOT model E2E)"
    }
} finally {Pop-Location}
