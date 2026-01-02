# Hopfield Networks Website

This is a Jekyll site using the Tactile theme for GitHub Pages.

## Prerequisites

This site requires Ruby 3.0 or higher. If you don't have a compatible Ruby version:

1. Install rbenv (Ruby version manager):

   ```bash
   brew install rbenv ruby-build
   ```

2. Initialize rbenv in your shell (add to `~/.zshrc`):

   ```bash
   eval "$(rbenv init - zsh)"
   ```

3. Install Ruby 3.3.6:
   ```bash
   rbenv install 3.3.6
   ```

## Setup

1. Navigate to the website directory:

   ```bash
   cd website
   ```

2. Set the Ruby version (if using rbenv):

   ```bash
   rbenv local 3.3.6
   ```

3. Install dependencies:

   ```bash
   gem install bundler
   bundle install
   ```

4. Build and serve locally:

   ```bash
   bundle exec jekyll serve
   ```

5. Visit `http://localhost:4000` in your browser.

## Structure

- `_config.yml` - Jekyll configuration with Tactile theme
- `index.md` - Homepage
- `hopfield.md` - Hopfield Networks essay page
- `lab_files/` - Images and assets referenced in the essay

## Pages

- Homepage: `/`
- Hopfield Networks Essay: `/hopfield.html`
