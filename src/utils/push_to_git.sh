#!/bin/bash

# Initialize git if not already
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
fi

# Check for git identity
if [ -z "$(git config user.email)" ]; then
    echo "Git identity not set."
    echo "Please configure it now for this repository."
    read -p "Enter your email: " git_email
    read -p "Enter your name: " git_name
    
    if [ -n "$git_email" ] && [ -n "$git_name" ]; then
        git config user.email "$git_email"
        git config user.name "$git_name"
    else
        echo "Using default identity for this backup."
        git config user.email "backup@example.com"
        git config user.name "Backup User"
    fi
fi

# Create .gitignore if it doesn't exist to avoid adding unwanted files inadvertently
if [ ! -f ".gitignore" ]; then
    echo "Creating .gitignore..."
    echo "__pycache__/" >> .gitignore
    echo "*.pyc" >> .gitignore
    echo ".DS_Store" >> .gitignore
    echo "data/shards/" >> .gitignore  # Don't commit shards
    echo "vox1_dev_wav/" >> .gitignore
    echo "vox1_test_wav/" >> .gitignore
    echo "*.zip" >> .gitignore
    echo "*.tar" >> .gitignore
    git add .gitignore
fi

# Add core files
echo "Adding code files..."
git add src config.yaml requirements.txt README.md

# Add metadata
if [ -d "data/meta" ]; then
    echo "Adding metadata..."
    git add data/meta
else
    echo "Warning: data/meta not found."
fi

# Status check before commit
if git diff --cached --quiet; then
    echo "No changes to commit."
else
    # Commit
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    git commit -m "Update: $timestamp"
    echo "Committed changes."
fi

# Check remote argument
remote_url="$1"

if [ -n "$remote_url" ]; then
    if git remote | grep -q "^origin$"; then
        echo "Remote 'origin' already exists. Updating URL..."
        git remote set-url origin "$remote_url"
    else
        echo "Adding remote 'origin'..."
        git remote add origin "$remote_url"
    fi
fi

# Push
current_branch=$(git branch --show-current)
if [ -z "$current_branch" ]; then
    current_branch="main"
    git branch -M main
fi

if git remote | grep -q "^origin$"; then
    echo "Pushing to origin/$current_branch..."
    git push -u origin "$current_branch"
else
    echo "---------------------------------------------------"
    echo "Remote 'origin' not configured."
    echo "Please run this script with your remote URL:"
    echo "  bash src/utils/push_to_git.sh <YOUR_REPO_URL>"
    echo "---------------------------------------------------"
fi
