using Documenter
using FrequencyDriftRateTransforms

makedocs(
    sitename = "FrequencyDriftRateTransforms", # Add ".jl"?
    format = Documenter.HTML(),
    modules = [FrequencyDriftRateTransforms],
    remotes = nothing,
    pages = [
        "Contents" => "index.md",
        "API" => "api.md",
        "Index" => "autoindex.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
