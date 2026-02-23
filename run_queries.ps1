$queries = [ordered]@{
    "powershell_1" = "How do I measure the execution time of a command?"
    "powershell_2" = "What is the syntax for creating a custom class?"
    "neovim_1" = "How can I map a key to toggle the spell checker?"
    "neovim_2" = "What is the recommended way to configure built-in LSP?"
    "fish_1" = "How do I create a function that runs on directory change?"
    "fish_2" = "How to iterate over a list of files?"
    "python_1" = "How to use concurrent.futures for parallel tasks?"
    "python_2" = "Differences between staticmethod and classmethod?"
}

$results = @{}

function Invoke-TimedRetrieval {
    param([string]$query, [string]$embedder)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $jsonOutput = doc-retrieval $query --embedder $embedder
    $sw.Stop()
    
    return @{
        TimeMs = $sw.Elapsed.TotalMilliseconds
        Output = $jsonOutput -join "`n"
    }
}

foreach ($key in $queries.Keys) {
    $q = $queries[$key]
    Write-Host "Running query: $q"
    $results[$key] = @{
        Query = $q
        pytorch = Invoke-TimedRetrieval -query $q -embedder "pytorch"
        gemini = Invoke-TimedRetrieval -query $q -embedder "gemini"
    }
}

$results | ConvertTo-Json -Depth 5 | Out-File run_results.json -Encoding utf8
Write-Host "Done!"