[CmdletBinding()]
param(
 [string]$Compiler='', [string]$Builtins='', [string]$Python='python', [string]$Device='A750',
 [ValidateSet('smoke','quick','full')][string]$Suite='full', [string]$Model='', [string]$CapsuleDir='',
 [ValidateRange(0,1000)][int]$Layer=0, [ValidateRange(2,10000)][int]$Repeats=28,
 [ValidateRange(0,128)][int]$ScrubMiB=0, [ValidateSet('4','8')][int]$TokenTile=4, [switch]$NoBuild, [switch]$Synthetic, [switch]$MoeOnly
)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
$Stamp=Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$Out=Join-Path $Root "build/results/$Stamp-v05-gemm-rc$TokenTile-$Suite"
$LogDir=Join-Path $Root 'logs'
New-Item -ItemType Directory -Force $LogDir | Out-Null
$Sources=0;if($Model){$Sources++};if($CapsuleDir){$Sources++};if($Synthetic){$Sources++}
if($Sources -gt 1){throw 'Choose exactly one of Model, CapsuleDir, Synthetic.'}
if($Suite -ne 'smoke' -and $Sources -eq 0){throw 'Provide -Model or -CapsuleDir. Use -Synthetic explicitly for synthetic-only tests.'}
New-Item -ItemType Directory -Force $Out | Out-Null
$Transcript=$false
try{
 Start-Transcript -Path (Join-Path $LogDir "gemm-$Stamp.log") | Out-Null;$Transcript=$true
 # v0.5 only compares GROUPED RC1 against RC4/8. MoeOnly remains an accepted alias.
 if(-not $NoBuild){& (Join-Path $PSScriptRoot 'build_windows.ps1') -Compiler $Compiler -Builtins $Builtins -GemmOnly}
 $Py=Get-Command $Python -ErrorAction SilentlyContinue
 if(-not $Py){throw 'Python 3.10+ required (standard library only).'}
 & $Py.Source -m unittest discover -s (Join-Path $Root 'tests') -p 'test_*.py'
 if($LASTEXITCODE -ne 0){throw 'Python tests failed.'}
 $RunArgs=@((Join-Path $Root 'tools/run_gemm_suite.py'),'--exe',(Join-Path $Root 'build/maple-gemm-compare.exe'),
  '--suite',$Suite,'--device',$Device,'--out',(Join-Path $Out 'gemm'),'--token-tile',"$TokenTile",'--repeats',"$Repeats",'--scrub-mib',"$ScrubMiB",'--layer',"$Layer")
 if($Model){$RunArgs+=@('--model',$Model)}
 if($CapsuleDir){$RunArgs+=@('--capsule-dir',$CapsuleDir)}
 if($Synthetic){$RunArgs+=@('--synthetic')}
 & $Py.Source @RunArgs
 if($LASTEXITCODE -ne 0){throw "GEMM suite failed. Preserve $Out."}
 Write-Host "Completed: $Out/gemm/q_speedup.csv"
 Write-Host 'Only multi-token DPAS added. GROUPED RC1 baseline; grouping + tile plan costs included. No server DLL/settings changed.'
}finally{if($Transcript){Stop-Transcript | Out-Null}}
