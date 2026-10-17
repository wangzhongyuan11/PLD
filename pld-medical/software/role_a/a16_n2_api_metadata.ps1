# A16-N2 步骤 0（可复现）：读取 48 张试点清单并用官方 API 取元数据（只读，不下载图像）
# 用法： powershell -NoProfile -ExecutionPolicy Bypass -File software/role_a/a16_n2_api_metadata.ps1
$ErrorActionPreference = 'Continue'
Set-Location -LiteralPath (Resolve-Path "$PSScriptRoot\..\..").Path
$run = 'runs\role_a_a16_novice_image_pilot'
$pilot = 'runs\role_a_a16_novice_mask_audit\pilot_selection.csv'
New-Item -ItemType Directory -Force -Path "$run\logs" | Out-Null
$rows = Import-Csv -LiteralPath $pilot
$out = @(); $fail = @(); $total = 0
foreach ($r in $rows) {
    $id = $r.isic_id
    $json = curl.exe -sS --max-time 30 --retry 1 --retry-delay 2 "https://api.isic-archive.com/api/v2/images/$id/" 2>&1
    if ($LASTEXITCODE -ne 0 -or -not $json) { $fail += "$id exit=$LASTEXITCODE"; continue }
    try { $j = ($json -join '') | ConvertFrom-Json } catch { $fail += "$id json_error"; continue }
    $full = $j.files.full
    $out += [pscustomobject]@{ isic_id = $id; area_band = $r.area_band; area_ratio = $r.area_ratio
        full_url = $full.url; full_size = [int64]$full.size; license = $j.copyright_license
        attribution = $j.attribution; image_type = $j.image_type }
    $total += [int64]$full.size
}
$peak = $total + $out.Count * 150000 + 20000000
$meta = [ordered]@{ schema = 'a16-n2-pilot-api-v1'
    pilot_selection_sha256 = (Get-FileHash $pilot -Algorithm SHA256).Hash.ToLower()
    count = $out.Count; failed = $fail; images = $out; sum_full_size_bytes = $total
    estimated_peak_bytes = $peak
    budget = @{ persistent_target = 250000000; temp_peak_target = 450000000 } }
$meta | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$run\api_metadata.json" -Encoding UTF8
Write-Output ("done count=$($out.Count) failed=$($fail.Count) bytes=$total")