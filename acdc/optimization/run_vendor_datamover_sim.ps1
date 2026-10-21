param(
    [Parameter(Mandatory=$true)][string]$Build,
    [Parameter(Mandatory=$true)][string]$Output,
    [string]$VivadoBin='D:/vivado/2026.1/Vivado/bin'
)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$buildPath=(Resolve-Path -LiteralPath $Build).Path
$outPath=[IO.Path]::GetFullPath($Output)
if(Test-Path -LiteralPath $outPath){throw 'Choose a new vendor simulation output directory'}
$ip='sources_1/bd/dma_top/ip/dma_top_datamover_0'
$netlist=Join-Path $buildPath "project/spatial.gen/$ip/dma_top_datamover_0_sim_netlist.v"
$configuration=Join-Path $buildPath "project/spatial.srcs/$ip/dma_top_datamover_0.xci"
$monitor=Join-Path $repo 'acdc/optimization/rtl/dma_transfer.sv'
$tb=Join-Path $repo 'acdc/optimization/sim/tb_vendor_datamover.sv'
$global=Join-Path $VivadoBin '../data/verilog/src/glbl.v'
$hashes=@{}
foreach($path in @($netlist,$configuration,$monitor,$tb,$global)){
    $hashes[$path]=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}
New-Item -ItemType Directory -Path $outPath | Out-Null
Push-Location $outPath
try{
    & "$VivadoBin/xvlog.bat" -sv -nolog $monitor $tb $netlist $global *> compile.log
    if($LASTEXITCODE -ne 0){throw 'Vendor DataMover compilation failed'}
    & "$VivadoBin/xelab.bat" -nolog tb_vendor_datamover glbl -L unisims_ver -L secureip -s vendor_datamover -timescale 1ns/1ps *> elaborate.log
    if($LASTEXITCODE -ne 0){throw 'Vendor DataMover elaboration failed'}
    & "$VivadoBin/xsim.bat" vendor_datamover -R *> simulation.log
    if($LASTEXITCODE -ne 0){throw 'Vendor DataMover simulation failed'}
    if(-not(Select-String -LiteralPath simulation.log -Pattern 'TB_PASS vendor_datamover' -Quiet)){
        throw 'Missing vendor integration self-check completion'
    }
    foreach($path in $hashes.Keys){
        if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $hashes[$path]){throw "Source drift: $path"}
    }
    @{schema='pld-acdc-vendor-datamover-simulation-v1';passed=$true;sources=$hashes;
      log_sha256=(Get-FileHash simulation.log -Algorithm SHA256).Hash.ToLowerInvariant();
      scope='actual generated AMD DataMover functional netlist and B DMA monitors with stalled AXI RAM; not PS7/DDR physical timing'} |
        ConvertTo-Json -Depth 5 | Set-Content simulation_manifest.json -Encoding utf8
    Select-String -LiteralPath simulation.log -Pattern 'VENDOR_PAIR|TB_PASS' | ForEach-Object {$_.Line}
}finally{Pop-Location}
