[CmdletBinding()]
param([string]$Compiler='', [string]$Device='A750', [int]$Tokens=1,
      [switch]$MapleShape, [switch]$DenseA8, [switch]$BuildOnly,
      [ValidateSet('direct','grouped','auto')][string]$Schedule='direct')
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
# Preserve user's merged source. This builds those exact sources, not supplied old .objs.
& (Join-Path $PSScriptRoot 'build_windows.ps1') -Compiler $Compiler -GroupingOnly
if(-not $Compiler){foreach($N in @('icpx.exe','icx.exe')){$C=Get-Command $N -ErrorAction SilentlyContinue;if($C){$Compiler=$C.Source;break}}}
if(-not $Compiler){throw 'Initialize oneAPI + MSVC before building Level4.'}
$F=@('-fexceptions','-fcxx-exceptions','-std=c++17','-fsycl','-O3','-fno-fast-math','-fsycl-device-code-split=per_kernel','-Iinclude')
$Extra=@();$R=& $Compiler '-print-resource-dir';if($LASTEXITCODE -eq 0 -and $R){$ResourceDir=([string](@($R)[-1])).Trim();$Candidate=Join-Path $ResourceDir 'lib/windows/clang_rt.builtins-x86_64.lib';if(Test-Path $Candidate){$Extra+=@($Candidate)}}
function RunLogged([string[]]$A,[string]$Log){
    $Saved=$ErrorActionPreference
    try{$ErrorActionPreference='Continue';& $Compiler @A 2>&1|Tee-Object -FilePath $Log;$rc=$LASTEXITCODE}finally{$ErrorActionPreference=$Saved}
    if($rc -ne 0){throw "Level4 build failed ($rc); preserve $Log. No server DLL changed."}
}
Push-Location $Root
try {
    RunLogged ($F+@('-c','src/maple_level4.cpp','-o','build/maple_level4.obj')) 'build/level4-build.log'
    $O=@('maple_w2a16','maple_w2a8','maple_moe','expert_grouping','maple_w2a8_grouped','expert_tiles','maple_w2a8_gemm','maple_level4')|ForEach-Object{"build/$_.obj"}
    RunLogged ($F+@('tools/level4_probe.cpp')+$O+$Extra+@('-o','build/maple-level4-probe.exe')) 'build/level4-link.log'
    & python 'tools/stamp_level4_build.py' --root . --exe 'build/maple-level4-probe.exe'
    if($LASTEXITCODE -ne 0){throw 'Level4 build provenance stamp failed'}
    if(-not $BuildOnly) {
        $Out='build/level4-'+(Get-Date -Format 'yyyyMMdd-HHmmss')
        $A=@('--device',$Device,'--tokens',"$Tokens",'--out',$Out,'--schedule',$Schedule)
        if($MapleShape){$A+=@('--maple')};if($DenseA8){$A+=@('--dense-a8')}
        & 'build/maple-level4-probe.exe' @A
        if($LASTEXITCODE -ne 0){throw "Level4 GPU test failed; preserve $Out."}
        & python 'tools/analyze_level4_trace.py' $Out
        if($LASTEXITCODE -ne 0){throw 'Level4 trace analysis failed'}
    }
} finally {Pop-Location}
