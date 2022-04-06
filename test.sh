# If there's issue with running the script, try:
# bundle clean --force

echo "Test page locally"
bundle install
jekyll build
jekyll serve # --livereload
