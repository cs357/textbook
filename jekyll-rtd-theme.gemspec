Gem::Specification.new do |spec|
  spec.name          = "jekyll-rtd-theme"
  spec.version       = "2.0.10-cs357"
  spec.authors       = ["mfsilva"]
  spec.email         = ["mfsilva@illinois.edu"]

  spec.summary       = "Fork of rundocs/jekyll-rtd-theme"
  spec.license       = "MIT"
  spec.homepage      = "https://github.com/cs357/jekyll-rtd-theme"

  spec.files         = `git ls-files -z`.split("\x0").select { |f| f.match(%r!^(assets|_layouts|_includes|_sass|LICENSE|README)!i) }

  spec.add_runtime_dependency "github-pages", "~> 209"
end
