using Documenter
using DopplerDriftSearch

makedocs(
    sitename = "DopplerDriftSearch",
    format = Documenter.HTML(),
    modules = [DopplerDriftSearch],
    remotes = nothing
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
