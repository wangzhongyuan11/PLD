$ErrorActionPreference = 'Stop'
$projectPath = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Push-Location -LiteralPath $projectPath
try {
    $rtlPath = 'hardware/rtl/learning_frame_threshold.v'
    & 'C:\iverilog\bin\iverilog.exe' -g2012 -s tb_learning_frame_threshold -o runs/learning_frame_default_check.vvp $rtlPath hardware/sim/tb_learning_frame_threshold.sv
    if ($LASTEXITCODE -ne 0) { throw 'Default compilation failed' }
    & 'C:\iverilog\bin\vvp.exe' runs/learning_frame_default_check.vvp | Tee-Object -FilePath runs/learning_frame_default_check.log
    if ($LASTEXITCODE -ne 0) { throw 'Default simulation failed' }
    foreach ($setting in @(@(12,100), @(12,80), @(1,0), @(1,255), @(65536,255))) {
        $frameSize = $setting[0]
        $thresholdValue = $setting[1]
        $artifactStem = "runs/learning_frame_n${frameSize}_t${thresholdValue}"
        & 'C:\iverilog\bin\iverilog.exe' -g2012 -s tb_learning_frame_parameters "-Ptb_learning_frame_parameters.N=$frameSize" "-Ptb_learning_frame_parameters.T=$thresholdValue" -o "$artifactStem.vvp" $rtlPath hardware/sim/tb_learning_frame_parameters.sv
        if ($LASTEXITCODE -ne 0) { throw "Compilation failed: $artifactStem" }
        & 'C:\iverilog\bin\vvp.exe' "$artifactStem.vvp" | Tee-Object -FilePath "$artifactStem.log"
        if ($LASTEXITCODE -ne 0) { throw "Simulation failed: $artifactStem" }
    }
} finally {
    Pop-Location
}
