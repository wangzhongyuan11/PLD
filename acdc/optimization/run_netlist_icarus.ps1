param(
    [Parameter(Mandatory=$true)][string]$IcarusBin,
    [Parameter(Mandatory=$true)][string]$Netlist,
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$ModelSha,
    [Parameter(Mandatory=$true)][string]$Output,
    [ValidateSet(4,6)][int]$P=6,
    [string]$VivadoRoot='D:/vivado/2026.1/Vivado'
)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$outputPath=[IO.Path]::GetFullPath($Output)
if(Test-Path -LiteralPath $outputPath){throw 'Choose a new simulation output directory'}
$ivl=[IO.Path]::GetFullPath((Join-Path $IcarusBin '../lib/ivl'))
$compiler=Join-Path $IcarusBin 'iverilog.exe'
$simulator=Join-Path $IcarusBin 'vvp.exe'
$python=Join-Path $repo 'acdc/.venv/Scripts/python.exe'
$netlistPath=(Resolve-Path -LiteralPath $Netlist).Path
$ddr=Join-Path $repo 'acdc/optimization/sim/datamover_memory_portable.sv'
$tb=Join-Path $repo 'acdc/optimization/sim/tb_executor_netlist.sv'
$global=Join-Path $VivadoRoot 'data/verilog/src/glbl.v'
$primitive=Join-Path $VivadoRoot 'data/verilog/src/unisims'
foreach($path in @($compiler,$simulator,$netlistPath,$ddr,$tb,$global)){
    if(-not(Test-Path -LiteralPath $path)){throw "Missing input: $path"}
}
$priorPath=$env:PATH
try{
    $env:PATH="$IcarusBin;$priorPath"
    Push-Location $repo
    try{
        & $python -m acdc.optimization.gen_executor_vectors $outputPath --p $P --divisor 8 --model $Model --model-spec-sha $ModelSha
        if($LASTEXITCODE -ne 0){throw 'Fixture generation failed'}
    }finally{Pop-Location}
    $inputs=@($netlistPath,$ddr,$tb,$global)
    $hashes=@{}
    foreach($path in $inputs){$hashes[$path]=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
    # Bind vendor primitive sources too; they are read from the user's Vivado installation.
    Get-ChildItem -LiteralPath $primitive -Filter '*.v' | ForEach-Object {
        $hashes[$_.FullName]=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    Push-Location $outputPath
    try{
        $watch=[Diagnostics.Stopwatch]::StartNew()
        & $compiler -B $ivl -g2012 -DACDC_PORTABLE_DDR -s tb_executor_netlist -s glbl -y $primitive -o netlist.vvp $netlistPath $ddr $tb $global *> compile.log
        if($LASTEXITCODE -ne 0){throw 'Icarus compilation failed; inspect compile.log'}
        $compileSeconds=$watch.Elapsed.TotalSeconds;$watch.Restart()
        & $simulator -i -M $ivl netlist.vvp *> simulation.log
        if($LASTEXITCODE -ne 0){throw 'Netlist simulation failed; inspect simulation.log'}
        $simulationSeconds=$watch.Elapsed.TotalSeconds
        if(-not(Select-String -LiteralPath simulation.log -Pattern 'TB_PASS executor_functional_netlist' -Quiet)){
            throw 'Missing self-checking completion marker'
        }
        foreach($path in $hashes.Keys){
            if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $hashes[$path]){throw "Source drift: $path"}
        }
        $report=@{schema='pld-acdc-icarus-netlist-v1';passed=$true;spatial_p=$P;spatial_divisor=8;
            model_spec_sha256=$ModelSha;sources=$hashes;compile_seconds=$compileSeconds;simulation_seconds=$simulationSeconds;
            log_sha256=(Get-FileHash -LiteralPath simulation.log -Algorithm SHA256).Hash.ToLowerInvariant();
            scope='functional synthesized executor, reduced grid, original vendor primitives; not SDF timing or full SoC simulation'}
        $report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath simulation_manifest.json -Encoding utf8
        Write-Output "NETLIST_PORTABLE_PASS P=$P output=$outputPath"
    }finally{Pop-Location}
}finally{$env:PATH=$priorPath}
