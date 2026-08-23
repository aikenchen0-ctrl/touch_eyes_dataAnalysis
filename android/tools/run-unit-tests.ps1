$ErrorActionPreference = "Stop"

$sourceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\")).Path
$linkRoot = "C:\gradle-home\zhifa-android-test"
if (Test-Path -LiteralPath $linkRoot) {
    $item = Get-Item -LiteralPath $linkRoot -Force
    if ($item.LinkType -ne "Junction" -or $item.Target -ne $sourceRoot) {
        throw "Test path already exists and does not point to ${sourceRoot}: $linkRoot"
    }
} else {
    New-Item -ItemType Junction -Path $linkRoot -Target $sourceRoot | Out-Null
}

Push-Location $linkRoot
try {
    & java -classpath gradle\wrapper\gradle-wrapper.jar `
        org.gradle.wrapper.GradleWrapperMain `
        :aispect-core:testDebugUnitTest `
        --no-daemon --max-workers=1 --offline --console=plain
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
