# Start the training watchdog at a fixed wall-clock time, detached from whatever launched it.
#
# Sleeping a fixed number of seconds drifts: the process takes time to spawn, and a laptop that
# suspends stops counting. Waiting on a target timestamp instead means the run starts when it was
# asked to, or immediately if that moment has already passed.
param(
    [Parameter(Mandatory = $true)][string]$At,          # "HH:mm:ss" or any parseable datetime
    [string]$Run = "D:/Certus/runs_torch/train_20260912_211356",
    [string]$Python = "C:\Program Files\Python311\python.exe"
)

$target = [datetime]::Parse($At)
$log = Join-Path $Run "watchdog.log"

# Start-Process truncates what it redirects into, so keep whatever the previous watchdog wrote.
# tail -F follows a recreated file, so anything watching the log survives the rotation.
if (Test-Path $log) { Move-Item $log "$log.$(Get-Date -Format 'yyyyMMdd_HHmmss')" -Force }
Add-Content -Path (Join-Path $Run "watchdog.armed.log") -Encoding utf8 `
    -Value ("[{0:HH:mm:ss}] delayed start armed for {1:HH:mm:ss}" -f (Get-Date), $target)

while ((Get-Date) -lt $target) {
    $left = ($target - (Get-Date)).TotalSeconds
    Start-Sleep -Seconds ([Math]::Min(30, [Math]::Max(1, $left)))
}

Set-Location "D:\Certus\torch"
# Redirect through Start-Process rather than PowerShell's *>> operator. In Windows PowerShell 5.1
# that operator writes UTF-16, which silently broke every grep-based watch on this log: the file
# had started out as ASCII, so patterns still matched the older lines and the failure looked like
# "nothing is happening" rather than an error. Start-Process passes the child's bytes through.
$err = Join-Path $Run "watchdog.err.log"
$proc = Start-Process -FilePath $Python -ArgumentList "-u", "watchdog.py", "--run", $Run `
    -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
$proc.WaitForExit()
