param([Parameter(Mandatory=$true)][string]$JobsPath)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$jobs = Get-Content -LiteralPath $JobsPath -Raw -Encoding UTF8 | ConvertFrom-Json
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voiceName = 'Microsoft Huihui Desktop'
$available = @($synth.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name })
if ($available -notcontains $voiceName) { throw "Required voice not found: $voiceName" }
$synth.SelectVoice($voiceName)
$synth.Rate = 0
$synth.Volume = 100
$script:currentMarks = @()
$handler = [System.EventHandler[System.Speech.Synthesis.BookmarkReachedEventArgs]] {
    param($sender, $eventArgs)
    $script:currentMarks += [PSCustomObject]@{
        name = $eventArgs.Bookmark
        milliseconds = [double]$eventArgs.AudioPosition.TotalMilliseconds
    }
}
$synth.add_BookmarkReached($handler)
try {
    $index = 0
    foreach ($job in $jobs) {
        $index += 1
        $wavPath = [string]$job.wav_path
        $alignmentPath = [string]$job.alignment_path
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($wavPath)) | Out-Null
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($alignmentPath)) | Out-Null
        $script:currentMarks = @()
        $parts = @()
        $text = [string]$job.text
        for ($i = 0; $i -lt $text.Length; $i++) {
            $escaped = [Security.SecurityElement]::Escape([string]$text[$i])
            $parts += ('<mark name="c{0:D4}"/>{1}' -f $i, $escaped)
        }
        $parts += ('<mark name="c{0:D4}"/>' -f $text.Length)
        $ssml = '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zh-CN">' + ($parts -join '') + '</speak>'
        $synth.SetOutputToWaveFile($wavPath)
        $synth.SpeakSsml($ssml)
        $synth.SetOutputToNull()
        $payload = [ordered]@{
            utterance_key = [string]$job.utterance_key
            text = $text
            voice_name = $voiceName
            marks = $script:currentMarks
        } | ConvertTo-Json -Depth 5
        [IO.File]::WriteAllText($alignmentPath, $payload, (New-Object Text.UTF8Encoding($false)))
        if (($index % 50) -eq 0 -or $index -eq $jobs.Count) {
            Write-Output ("SYNTHESIZED {0}/{1}" -f $index, $jobs.Count)
        }
    }
} finally {
    $synth.remove_BookmarkReached($handler)
    $synth.Dispose()
}
