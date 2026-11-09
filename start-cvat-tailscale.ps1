$ErrorActionPreference = 'Stop'

$tailscale = 'C:\Program Files\Tailscale\tailscale.exe'
if (-not (Test-Path -LiteralPath $tailscale)) {
    throw 'Tailscale is not installed at C:\Program Files\Tailscale.'
}

$tailscaleIp = (& $tailscale ip -4 2>$null | Select-Object -First 1).Trim()
if ($LASTEXITCODE -ne 0 -or $tailscaleIp -notmatch '^100\.(?:[0-9]{1,3}\.){2}[0-9]{1,3}$') {
    throw 'Unable to read the Tailscale IPv4 address. Sign in to Tailscale and run this script again.'
}

$env:CVAT_HOST = $tailscaleIp
$env:DOCKER_CONFIG = (Join-Path $PSScriptRoot '.docker-deploy')
docker compose -f (Join-Path $PSScriptRoot 'docker-compose.yml') -f (Join-Path $PSScriptRoot 'docker-compose.dev.yml') -f (Join-Path $PSScriptRoot 'components/serverless/docker-compose.serverless.yml') up -d
Write-Host "CVAT is available at http://$tailscaleIp`:8080"
