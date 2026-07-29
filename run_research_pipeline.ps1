# Edit these paths before running.
$DataRoot = "D:/WORK/00-JPGU_2026/data/all_training_archive"
$Prepared = "D:/WORK/00-JPGU_2026/data/prepared_ssif_v3"
$Output = "D:/WORK/00-JPGU_2026/output/ssif_v3_seed20260728"
$Seed = 20260728

python prepare_ssif_dataset.py audit-split `
  --data-dir $DataRoot `
  --output-dir $Prepared `
  --label-horizon 120 `
  --min-label-valid-fraction 0.80 `
  --min-window-valid-fraction 0.80 `
  --train-ratio 0.70 `
  --validation-ratio 0.10 `
  --calibration-ratio 0.10 `
  --test-ratio 0.10 `
  --split-candidates 5000 `
  --seed $Seed

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Review the audit files in $Prepared before training."
Write-Host "After review, run the training command below manually:"
Write-Host "python train_ssif_v3.py train-all --data-dir $DataRoot --split-manifest $Prepared/split_manifest.json --output-dir $Output --windows 10 15 20 25 30 35 40 --label-horizon 120 --cohort common --epochs 30 --batch-size 16 --eval-batch-size 64 --lr 3e-4 --min-precision 0.90 --seed $Seed --window-seed-mode same --amp"
