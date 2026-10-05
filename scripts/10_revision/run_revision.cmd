@echo off
REM ==================================================================
REM  GRSL-01920-2026 revision experiments - cmd.exe one-command runner
REM  All paths hard-coded for this machine. Run from anywhere.
REM
REM  Usage:
REM    run_revision.cmd cleanup       remove the bogus "$WORK" folder from the failed run
REM    run_revision.cmd convert       step 1: dataset conversion (CPU, ~10 min)
REM    run_revision.cmd cache-ship    step 2: SAR-Ship val+test inference (CPU, 1-2 h)
REM    run_revision.cmd cache-sardet  step 2: SARDet val+test inference (CPU, 3-7 h, overnight OK)
REM    run_revision.cmd analyze       steps 3-7: manifests, proxy, main results, CV, k-sweep, transfer, tables
REM    run_revision.cmd cfar          step H: clutter-invariant false-alarm evaluation (minutes)
REM    run_revision.cmd figures       steps 8-9: Fig.2 curve export + Fig.3 inputs
REM    run_revision.cmd status        show how many .npz are cached so far
REM    run_revision.cmd all           convert + both caches + analyze
REM ==================================================================
setlocal
set "REPO=D:\Project\eat-sar\eat-sar\eat-sar"
set "WORK=D:\Project\eat-sar\work"
set "SSROOT=%REPO%\SAR-Ship-Dataset"
set "SDROOT=%REPO%\SARDet-100K"
set "W_SS=%REPO%\runs_backup\sarship\weights\best.pt"
set "W_SD=%REPO%\runs_backup\sardet100k\best.pt"
set "Y=%WORK%\yolo"
set "SS=%Y%\sarship"
set "SD=%Y%\sardet"
set "CP=%WORK%\cache_preds"
set "OUT=%REPO%\revision_out"

cd /d "%REPO%"
set "PYTHONPATH=%REPO%\src"
set "PYTHONIOENCODING=utf-8"

if "%~1"=="" goto :usage
if /i "%~1"=="cleanup"      goto :cleanup
if /i "%~1"=="convert"      goto :convert
if /i "%~1"=="cache-ship"   goto :cache-ship
if /i "%~1"=="cache-sardet" goto :cache-sardet
if /i "%~1"=="analyze"      goto :analyze
if /i "%~1"=="figures"      goto :figures
if /i "%~1"=="cfar"         goto :cfar
if /i "%~1"=="status"       goto :status
if /i "%~1"=="all"          goto :convert
goto :usage

REM ------------------------------------------------------------------
:cleanup
if exist "%REPO%\$WORK" rmdir /s /q "%REPO%\$WORK"
echo [ok] removed stray "$WORK" folder if it existed
goto :end

REM ------------------------------------------------------------------
:convert
if not exist "%WORK%" mkdir "%WORK%"
if not exist "%OUT%"  mkdir "%OUT%"
python scripts\10_revision\convert_datasets.py --sarship-root "%SSROOT%" --sardet-root "%SDROOT%" --out-root "%Y%"
if errorlevel 1 goto :fail
echo.
echo === class map (ship class id) ===
type "%SD%\class_map.json"
echo.
echo Next: run_revision.cmd cache-ship
if /i "%~1"=="all" goto :cache-ship
goto :end

REM ------------------------------------------------------------------
:cache-ship
if not exist "%SS%\val.txt" echo [error] run "run_revision.cmd convert" first & goto :end
python scripts\10_revision\cache_preds.py --weights "%W_SS%" --img-dir "%SSROOT%\JPEGImages" --list "%SS%\val.txt" --out-dir "%CP%\sarship_val" --device cpu
python scripts\10_revision\cache_preds.py --weights "%W_SS%" --img-dir "%SSROOT%\JPEGImages" --list "%SS%\test.txt" --out-dir "%CP%\sarship_test" --device cpu
echo Next: run_revision.cmd cache-sardet   (or analyze to test the chain on SAR-Ship only)
if /i "%~1"=="all" goto :cache-sardet
goto :end

REM ------------------------------------------------------------------
:cache-sardet
if not exist "%SD%\val.txt" echo [error] run "run_revision.cmd convert" first & goto :end
python scripts\10_revision\cache_preds.py --weights "%W_SD%" --img-dir "%SDROOT%\Images\val" --list "%SD%\val.txt" --out-dir "%CP%\sardet_val" --device cpu
python scripts\10_revision\cache_preds.py --weights "%W_SD%" --img-dir "%SDROOT%\Images\test" --list "%SD%\test.txt" --out-dir "%CP%\sardet_test" --device cpu
echo Next: run_revision.cmd analyze
if /i "%~1"=="all" goto :analyze
goto :end

REM ------------------------------------------------------------------
:analyze
if not exist "%SS%\labels\all" echo [error] run "run_revision.cmd convert" first & goto :end
if not exist "%CP%\sarship_test" echo [error] run "run_revision.cmd cache-ship" first & goto :end
if exist "%Y%\env.cmd" call "%Y%\env.cmd"
if not defined SHIP_ID set "SHIP_ID=0"
echo Using SHIP_ID=%SHIP_ID%  (from %Y%\env.cmd)
if not exist "%OUT%" mkdir "%OUT%"

echo --- [3/7] manifests ---
python scripts\10_revision\make_manifest.py --cache-dir "%CP%\sarship_val" --img-dir "%SSROOT%\JPEGImages" --list "%SS%\val.txt" --out "%CP%\sarship_val\manifest.csv"
python scripts\10_revision\make_manifest.py --cache-dir "%CP%\sarship_test" --img-dir "%SSROOT%\JPEGImages" --list "%SS%\test.txt" --out "%CP%\sarship_test\manifest.csv"
python scripts\10_revision\make_manifest.py --cache-dir "%CP%\sardet_val" --img-dir "%SDROOT%\Images\val" --list "%SD%\val.txt" --out "%CP%\sardet_val\manifest.csv"
python scripts\10_revision\make_manifest.py --cache-dir "%CP%\sardet_test" --img-dir "%SDROOT%\Images\test" --list "%SD%\test.txt" --out "%CP%\sardet_test\manifest.csv"

echo --- [4/7] clutter proxy recomputation ---
python scripts\10_revision\recompute_proxy.py --manifest "%CP%\sarship_val\manifest.csv" --label-root "%SS%\labels\all" --out-manifest "%CP%\sarship_val\manifest_proxy.csv" --k-list 0 1 2 3 5 8 --k-main 3 --mask-gate 0.25 --target-class %SHIP_ID% --report "%OUT%\sarship_val_proxy_report.json"
python scripts\10_revision\recompute_proxy.py --manifest "%CP%\sarship_test\manifest.csv" --label-root "%SS%\labels\all" --out-manifest "%CP%\sarship_test\manifest_proxy.csv" --k-list 0 1 2 3 5 8 --k-main 3 --mask-gate 0.25 --target-class %SHIP_ID% --report "%OUT%\sarship_test_proxy_report.json"
python scripts\10_revision\recompute_proxy.py --manifest "%CP%\sardet_val\manifest.csv" --label-root "%SD%\labels\val" --out-manifest "%CP%\sardet_val\manifest_proxy.csv" --k-list 0 3 5 8 12 16 --k-main 8 --mask-gate 0.25 --target-class %SHIP_ID% --report "%OUT%\sardet_val_proxy_report.json"
python scripts\10_revision\recompute_proxy.py --manifest "%CP%\sardet_test\manifest.csv" --label-root "%SD%\labels\test" --out-manifest "%CP%\sardet_test\manifest_proxy.csv" --k-list 0 3 5 8 12 16 --k-main 8 --mask-gate 0.25 --target-class %SHIP_ID% --report "%OUT%\sardet_test_proxy_report.json"

echo --- [5/7] main results + controls ---
python scripts\10_revision\run_main_eval.py --val-manifest "%CP%\sarship_val\manifest_proxy.csv" --val-labels "%SS%\labels\all" --test-manifest "%CP%\sarship_test\manifest_proxy.csv" --test-labels "%SS%\labels\all" --target-class %SHIP_ID% --baseline-T 0.40 --n-perm 20 --require-gt --out "%OUT%\sarship_main.json"
python scripts\10_revision\run_main_eval.py --val-manifest "%CP%\sardet_val\manifest_proxy.csv" --val-labels "%SD%\labels\val" --test-manifest "%CP%\sardet_test\manifest_proxy.csv" --test-labels "%SD%\labels\test" --target-class %SHIP_ID% --baseline-T 0.40 --n-perm 20 --require-gt --out "%OUT%\sardet_main.json"

echo --- [6/7] 10-fold CV, k-sweep, transfer ---
python scripts\10_revision\kfold_cv.py --val-manifest "%CP%\sarship_val\manifest_proxy.csv" --val-labels "%SS%\labels\all" --test-manifest "%CP%\sarship_test\manifest_proxy.csv" --test-labels "%SS%\labels\all" --target-class %SHIP_ID% --K 10 --require-gt --out "%OUT%\sarship_kfold.json"
python scripts\10_revision\kfold_cv.py --val-manifest "%CP%\sardet_val\manifest_proxy.csv" --val-labels "%SD%\labels\val" --test-manifest "%CP%\sardet_test\manifest_proxy.csv" --test-labels "%SD%\labels\test" --target-class %SHIP_ID% --K 10 --require-gt --out "%OUT%\sardet_kfold.json"
python scripts\10_revision\sweep_k.py --val-manifest "%CP%\sarship_val\manifest_proxy.csv" --val-labels "%SS%\labels\all" --test-manifest "%CP%\sarship_test\manifest_proxy.csv" --test-labels "%SS%\labels\all" --target-class %SHIP_ID% --k-main 3 --require-gt --out "%OUT%\sarship_ksweep.json"
python scripts\10_revision\sweep_k.py --val-manifest "%CP%\sardet_val\manifest_proxy.csv" --val-labels "%SD%\labels\val" --test-manifest "%CP%\sardet_test\manifest_proxy.csv" --test-labels "%SD%\labels\test" --target-class %SHIP_ID% --k-main 8 --require-gt --out "%OUT%\sardet_ksweep.json"
python scripts\10_revision\cross_transfer.py --src-main "%OUT%\sardet_main.json" --tgt-main "%OUT%\sarship_main.json" --tgt-test-manifest "%CP%\sarship_test\manifest_proxy.csv" --tgt-test-labels "%SS%\labels\all" --target-class %SHIP_ID% --require-gt --out "%OUT%\transfer_sardet_to_sarship.json"
python scripts\10_revision\cross_transfer.py --src-main "%OUT%\sarship_main.json" --tgt-main "%OUT%\sardet_main.json" --tgt-test-manifest "%CP%\sardet_test\manifest_proxy.csv" --tgt-test-labels "%SD%\labels\test" --target-class %SHIP_ID% --require-gt --out "%OUT%\transfer_sarship_to_sardet.json"

echo --- [7/7] summary tables ---
python scripts\10_revision\make_tables.py --out-dir "%OUT%"
echo.
echo DONE. Send back the contents of:
echo   %Y%\conversion_summary.json
echo   %OUT%\numbers.json
echo   %OUT%\table_rows.tex
goto :end

REM ------------------------------------------------------------------
:cfar
if not exist "%CP%\sarship_test\manifest_proxy.csv" echo [error] run "run_revision.cmd analyze" first & goto :end
if exist "%Y%\env.cmd" call "%Y%\env.cmd"
if not defined SHIP_ID set "SHIP_ID=0"
if not exist "%OUT%" mkdir "%OUT%"
echo --- [H] CFAR-style evaluation: SAR-Ship ---
python scripts\10_revision\cfar_eval.py --val-manifest "%CP%\sarship_val\manifest_proxy.csv" --val-labels "%SS%\labels\all" --test-manifest "%CP%\sarship_test\manifest_proxy.csv" --test-labels "%SS%\labels\all" --target-class %SHIP_ID% --require-gt --n-perm 20 --n-boot 200 --out "%OUT%\sarship_cfar.json" > "%OUT%\sarship_cfar.txt" 2>&1
type "%OUT%\sarship_cfar.txt" | findstr /v /c:"load"
echo --- [H] CFAR-style evaluation: SARDet-100K ---
python scripts\10_revision\cfar_eval.py --val-manifest "%CP%\sardet_val\manifest_proxy.csv" --val-labels "%SD%\labels\val" --test-manifest "%CP%\sardet_test\manifest_proxy.csv" --test-labels "%SD%\labels\test" --target-class %SHIP_ID% --require-gt --n-perm 20 --n-boot 200 --out "%OUT%\sardet_cfar.json" > "%OUT%\sardet_cfar.txt" 2>&1
type "%OUT%\sardet_cfar.txt" | findstr /v /c:"load"
echo.
echo DONE. Send back: %OUT%\sarship_cfar.json  and  %OUT%\sardet_cfar.json
goto :end

REM ------------------------------------------------------------------
:figures
if not exist "%OUT%\numbers.json" echo [error] run "run_revision.cmd analyze" first & goto :end
echo --- [8] Fig.2 curve data exports ---
python scripts\10_revision\export_curves.py --numbers "%OUT%\numbers.json" --dataset sarship --test-manifest "%CP%\sarship_test\manifest_proxy.csv" --test-labels "%SS%\labels\all" --target-class %SHIP_ID% --out "%OUT%\curves_sarship.json"
python scripts\10_revision\export_curves.py --numbers "%OUT%\numbers.json" --dataset sardet --test-manifest "%CP%\sardet_test\manifest_proxy.csv" --test-labels "%SD%\labels\test" --target-class %SHIP_ID% --out "%OUT%\curves_sardet.json"
echo --- [9] Fig.3 case-figure inputs ---
python scripts\10_revision\prep_case_inputs.py --numbers "%OUT%\numbers.json" --dataset sarship --test-manifest "%CP%\sarship_test\manifest_proxy.csv" --label-root "%SS%" --label-split all --out-dir "%OUT%"
python scripts\10_revision\prep_case_inputs.py --numbers "%OUT%\numbers.json" --dataset sardet --test-manifest "%CP%\sardet_test\manifest_proxy.csv" --label-root "%SD%" --label-split test --out-dir "%OUT%"
echo Then run the two plot_case_examples_v2.py commands printed above (needs scripts\07_viz).
echo Send back: %OUT%\curves_sarship.json, %OUT%\curves_sardet.json, and both case_examples PDFs.
goto :end

REM ------------------------------------------------------------------
:status
echo npz counts so far:
for %%D in (sarship_val sarship_test sardet_val sardet_test) do (
  if exist "%CP%\%%D" (
    echo|set /p="%%D: "
    dir /b "%CP%\%%D\*.npz" 2>nul | find /c /v ""
  ) else (
    echo %%D: not started
  )
)
goto :end

REM ------------------------------------------------------------------
:usage
echo Usage: run_revision.cmd { cleanup ^| convert ^| cache-ship ^| cache-sardet ^| analyze ^| status ^| all }
goto :end

:fail
echo [error] the last command failed - scroll up for the Python traceback
goto :end

:end
endlocal
