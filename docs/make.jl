using Documenter, MCDTS


makedocs(sitename="MCDTS", modules=[MCDTS], doctest=true,
pages = [
    "Home" => "index.md",
    "Reference" => "ref.md"
    ]
    )
