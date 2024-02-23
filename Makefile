DEBUG=JEKYLL_GITHUB_TOKEN=blank PAGES_API_URL=http://0.0.0.0
ALIAS=jekyll-rtd-theme

help:
	@echo "HomePage: https://github.com/rundocs/${ALIAS}\n"
	@echo "Usage:"
	@echo "    make [subcommand]\n"
	@echo "Subcommands:"
	@echo "    install   Install the theme dependencies"
	@echo "    format    Format all files"
	@echo "    report    Make a report from Google lighthouse"
	@echo "    clean     Clean the workspace"
	@echo "    dist      Build the theme css and script"
	@echo "    status    Display status before push"
	@echo "    build     Build the test site"
	@echo "    server    Make a livereload jekyll server to development"
	@echo "    checkout  Reset the theme minified css and script to last commit"

checkout:
	@git checkout _config.yml
	@git checkout assets/js/theme.min.js
	@git checkout assets/css/theme.min.css

install:
	@npm install -g yarn
	@cd cs357-rtd-theme && yarn install
	@bundle install

format:
	@cd cs357-rtd-theme && yarn format

report:
	@cd cs357-rtd-theme && yarn report

clean:
	@bundle exec jekyll clean

dist: format clean
	@cd cs357-rtd-theme &&  yarn build

status: format clean checkout
	@git status

build: dist
	@${DEBUG} JEKYLL_ENV=production bundle exec jekyll build --safe --profile

server: dist
	@${DEBUG} bundle exec jekyll server --safe --livereload