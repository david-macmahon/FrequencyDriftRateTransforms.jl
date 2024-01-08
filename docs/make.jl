using Documenter
using DopplerDriftSearch

makedocs(
    sitename = "DopplerDriftSearch",
    format = Documenter.HTML(),
    modules = [DopplerDriftSearch]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
