# Scheduled props pulls and pre-kickoff refreshes (2026-11-11, audits B6 and B10; docs/audit_leads_2026-11-10.md).
# Registers Windows Task Scheduler tasks under \NFLScholar\, for this user, run only while logged on (no stored password),
# windowless (pythonw), each limited to 30 minutes. A run missed while the PC was off starts when it is next on: a late
# close pull only adds a line after kickoff, which the record ignores (it reads lines pulled before kickoff).
# Logs: .sweeps\scheduled\pull_props.log and refresh_pre_kickoff.log.
#
# Times are Central (this PC's zone); kickoffs are Eastern minus one hour:
#   pulls (free books)    Tue 09:00, Thu 11:00, Fri 17:00, Sat 17:00, Sun 08:15 (international games)
#   Odds API (credits)    Sun 10:30 only, through scripts/pull_props.py --odds-api (data/odds_api_props.py budget rules)
#   refresh (B10)         75 min before each window: Thu 18:00, Sun 07:15 (international), Sun 10:45, Sun 13:50,
#                         Sun 18:05, Mon 18:00
#   close pulls           15 min before each window: Thu 19:00, Sun 11:45, Sun 14:50, Sun 19:05, Mon 19:00
# Saturday and holiday games late in the season are not covered; add a one-off with Add-Pull below if wanted.
#
#   powershell -ExecutionPolicy Bypass -File scripts\schedule_tasks.ps1            # register (or update) all
#   powershell -ExecutionPolicy Bypass -File scripts\schedule_tasks.ps1 -Remove    # remove all
#   powershell -ExecutionPolicy Bypass -File scripts\schedule_tasks.ps1 -List
param([switch]$Remove, [switch]$List)

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$py = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\pythonw.exe'
$folder = '\NFLScholar\'

if ($List) {
    Get-ScheduledTask -TaskPath $folder -ErrorAction SilentlyContinue | ForEach-Object {
        $i = $_ | Get-ScheduledTaskInfo
        '{0,-28} next {1}  last {2} (result {3})' -f $_.TaskName, $i.NextRunTime, $i.LastRunTime, $i.LastTaskResult
    }
    return
}
if ($Remove) {
    Get-ScheduledTask -TaskPath $folder -ErrorAction SilentlyContinue | Unregister-ScheduledTask -Confirm:$false
    'removed every task under ' + $folder
    return
}
if (-not (Test-Path $py)) { throw "pythonw not found at $py" }

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

function Add-Task($name, $day, $time, $script) {
    $action = New-ScheduledTaskAction -Execute $py -Argument $script -WorkingDirectory $root
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $day -At $time
    Register-ScheduledTask -TaskPath $folder -TaskName $name -Action $action -Trigger $trigger -Settings $settings `
        -Principal $principal -Description "NFL Scholar ($script)" -Force | Out-Null
    '{0,-28} {1} {2}  {3}' -f $name, $day, $time, $script
}
function Add-Pull($name, $day, $time, $extra = '') { Add-Task $name $day $time ("scripts\pull_props.py $extra").Trim() }
function Add-Refresh($name, $day, $time) { Add-Task $name $day $time 'scripts\refresh_pre_kickoff.py' }

Add-Pull 'pull tue' Tuesday '09:00'
Add-Pull 'pull thu' Thursday '11:00'
Add-Pull 'pull fri' Friday '17:00'
Add-Pull 'pull sat' Saturday '17:00'
Add-Pull 'pull sun intl' Sunday '08:15'
Add-Pull 'pull sun odds api' Sunday '10:30' '--odds-api'
Add-Refresh 'refresh thu' Thursday '18:00'
Add-Refresh 'refresh sun intl' Sunday '07:15'
Add-Refresh 'refresh sun early' Sunday '10:45'
Add-Refresh 'refresh sun late' Sunday '13:50'
Add-Refresh 'refresh sun night' Sunday '18:05'
Add-Refresh 'refresh mon' Monday '18:00'
Add-Pull 'close thu' Thursday '19:00'
Add-Pull 'close sun early' Sunday '11:45'
Add-Pull 'close sun late' Sunday '14:50'
Add-Pull 'close sun night' Sunday '19:05'
Add-Pull 'close mon' Monday '19:00'
