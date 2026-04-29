$url = "http://localhost:8080/status.php"
$maxRetries = 40
$retryCount = 0
$isReady = $false

while ($retryCount -lt $maxRetries) {
    try {
        $response = Invoke-WebRequest -Uri $url -Method Get -TimeoutSec 5 -ErrorAction Stop
        if ($response.Content -match 'installed.:.true') {
            Write-Host "Nextcloud is ready"
            $isReady = $true
            break
        }
    } catch {
        Write-Host "Waiting..."
    }
    $retryCount++
    Start-Sleep -Seconds 10
}

if ($isReady) {
    $users = @("user1", "user2", "user3")
    $password = "majorpro_123"
    foreach ($u in $users) {
        Write-Host "Creating $u"
        docker exec --user www-data -e OC_PASS=$password app php occ user:add --password-from-env --display-name=$u $u
    }
}
else {
    Write-Host "Failed to start"
    exit 1
}
