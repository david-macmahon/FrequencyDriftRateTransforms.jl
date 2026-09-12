using Documenter
using FrequencyDriftRateTransforms

makedocs(
    sitename = "FrequencyDriftRateTransforms.jl",
    format = Documenter.HTML(),
    modules = [FrequencyDriftRateTransforms],
    repo = Documenter.Remotes.GitHub("david-macmahon", "FrequencyDriftRateTransforms.jl"),
    pages = [
        "Contents" => "index.md",
        "API" => "api.md",
        "Index" => "autoindex.md"
    ]
)

deploydocs(
    repo = "github.com/david-macmahon/FrequencyDriftRateTransforms.jl.git",
    devbranch = "main",
    push_preview = true,
)
