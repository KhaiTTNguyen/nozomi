#!/bin/bash

# NOZOMI Version Management Script
# This script helps create properly tagged versions following academic best practices

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  current     Show current version information"
    echo "  tag         Create a new version tag"
    echo "  prepare     Prepare files for version release"
    echo "  zenodo      Generate information for Zenodo DOI"
    echo "  citation    Generate citation information"
    echo ""
    echo "Options for 'tag':"
    echo "  -v, --version VERSION   Specify version (e.g., 1.0.0)"
    echo "  -m, --message MESSAGE   Tag annotation message"
    echo "  --dry-run              Show what would be done without executing"
    echo ""
    echo "Examples:"
    echo "  $0 current                           # Show current version info"
    echo "  $0 tag -v 1.1.0 -m \"Bug fixes\"      # Create tagged version"
    echo "  $0 prepare                          # Prepare release files"
}

check_git_status() {
    if ! git diff-index --quiet HEAD --; then
        echo -e "${RED}Error: Working directory has uncommitted changes.${NC}"
        echo "Please commit all changes before creating a version tag."
        exit 1
    fi
    
    if ! git diff-index --quiet --cached HEAD --; then
        echo -e "${RED}Error: Index has staged changes.${NC}"
        echo "Please commit all staged changes before creating a version tag."
        exit 1
    fi
}

get_current_version() {
    if [ -f "VERSION" ]; then
        cat VERSION
    else
        echo "0.0.0"
    fi
}

get_git_hash() {
    git rev-parse HEAD 2>/dev/null || echo "unknown"
}

get_git_tag() {
    git describe --tags --exact-match 2>/dev/null || echo ""
}

show_current_version() {
    echo -e "${BLUE}=== NOZOMI Version Information ===${NC}"
    echo ""
    
    local version=$(get_current_version)
    local git_hash=$(get_git_hash)
    local git_tag=$(get_git_tag)
    
    echo "Version file: $version"
    echo "Git hash: $git_hash"
    
    if [ -n "$git_tag" ]; then
        echo "Git tag: $git_tag"
        echo -e "${GREEN}✓ This commit is tagged as version $git_tag${NC}"
    else
        echo -e "${YELLOW}⚠ This commit is not tagged${NC}"
    fi
    
    echo ""
    echo "Full citation hash: $git_hash"
    echo "Short hash: ${git_hash:0:7}"
    
    # Check if Python can import and get version
    if command -v python3 &> /dev/null; then
        echo ""
        echo -e "${BLUE}Python version info:${NC}"
        python3 -c "import sys; sys.path.insert(0, '.'); import nozomi; print(f'Package version: {nozomi.get_version()}'); print(f'Git hash: {nozomi.__git_hash__}')" 2>/dev/null || echo "Could not import nozomi package"
    fi
}

validate_version() {
    local version="$1"
    if [[ ! $version =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?$ ]]; then
        echo -e "${RED}Error: Invalid version format '$version'${NC}"
        echo "Version must follow semantic versioning (e.g., 1.0.0, 1.2.3-beta)"
        return 1
    fi
}

update_version_file() {
    local new_version="$1"
    echo "Updating VERSION file to $new_version"
    echo "$new_version" > VERSION
}

update_init_version() {
    local new_version="$1"
    echo "Updating __init__.py version to $new_version"
    
    # Update the version in __init__.py
    sed -i "s/__version__ = \".*\"/__version__ = \"$new_version\"/" __init__.py
}

create_git_hash_file() {
    local git_hash="$1"
    echo "Creating .git_hash file for distributed versions"
    echo "$git_hash" > .git_hash
}

create_version_tag() {
    local version="$1"
    local message="$2"
    local dry_run="$3"
    
    if [ -z "$version" ]; then
        echo -e "${RED}Error: Version is required${NC}"
        print_usage
        exit 1
    fi
    
    validate_version "$version"
    
    if [ -z "$message" ]; then
        message="Release version $version"
    fi
    
    echo -e "${BLUE}=== Creating Version Tag ===${NC}"
    echo "Version: $version"
    echo "Message: $message"
    echo ""
    
    if [ "$dry_run" = "true" ]; then
        echo -e "${YELLOW}DRY RUN - Would execute:${NC}"
        echo "1. Update VERSION file to $version"
        echo "2. Update __init__.py version to $version"
        echo "3. Create .git_hash file"
        echo "4. git add VERSION __init__.py .git_hash"
        echo "5. git commit -m \"Version $version\""
        echo "6. git tag -a v$version -m \"$message\""
        return
    fi
    
    check_git_status
    
    # Check if tag already exists
    if git rev-parse "v$version" >/dev/null 2>&1; then
        echo -e "${RED}Error: Tag v$version already exists${NC}"
        exit 1
    fi
    
    local current_hash=$(get_git_hash)
    
    # Update version files
    update_version_file "$version"
    update_init_version "$version"
    create_git_hash_file "$current_hash"
    
    # Commit version update
    git add VERSION __init__.py .git_hash
    git commit -m "Version $version"
    
    # Create annotated tag
    git tag -a "v$version" -m "$message"
    
    echo -e "${GREEN}✓ Successfully created version tag v$version${NC}"
    echo ""
    echo "To push the tag to remote:"
    echo "  git push origin v$version"
    echo "  git push origin master"  # or your main branch
}

prepare_release() {
    echo -e "${BLUE}=== Preparing Release Files ===${NC}"
    
    local version=$(get_current_version)
    local git_hash=$(get_git_hash)
    
    # Create release directory
    local release_dir="release_v$version"
    mkdir -p "$release_dir"
    
    # Copy important files
    echo "Creating release package in $release_dir/"
    
    # Create a comprehensive README for the release
    cat > "$release_dir/RELEASE_INFO.md" << EOF
# NOZOMI Release v$version

**Version**: $version
**Git Hash**: $git_hash
**Date**: $(date -u +"%Y-%m-%dT%H:%M:%SZ")

## Citation Information

For academic publications, please cite this specific version:

### Text Citation
NOZOMI Development Team. NOZOMI: 3D white matter substrate generation and Monte Carlo diffusion simulation toolkit. Version $version. Git hash: ${git_hash:0:7}. Available at: https://github.com/your-username/nozomi

### BibTeX Entry
\`\`\`bibtex
@software{nozomi_${version//./\_},
  author = {NOZOMI Development Team},
  title = {NOZOMI: 3D White Matter Substrate Generation and Monte Carlo Diffusion Simulation},
  version = {$version},
  url = {https://github.com/your-username/nozomi},
  note = {Git hash: ${git_hash:0:7}},
  doi = {[Assign via Zenodo]}
}
\`\`\`

## Reproducibility Information

**Complete Git Hash**: \`$git_hash\`

This hash uniquely identifies the exact state of the code used. To reproduce results:

1. Clone the repository: \`git clone https://github.com/your-username/nozomi.git\`
2. Checkout this exact version: \`git checkout $git_hash\`
3. Follow installation instructions in README.md

## Files Included

- Source code snapshot
- Documentation
- Example configurations
- Validation tests

## Zenodo Upload Checklist

- [ ] Upload to Zenodo
- [ ] Get DOI
- [ ] Update citations with DOI
- [ ] Tag GitHub release with DOI

EOF

    echo "Release information written to $release_dir/RELEASE_INFO.md"
    echo -e "${GREEN}✓ Release preparation complete${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Review $release_dir/RELEASE_INFO.md"
    echo "2. Upload to Zenodo for DOI assignment"
    echo "3. Update GitHub release with DOI"
}

generate_zenodo_info() {
    echo -e "${BLUE}=== Zenodo Upload Information ===${NC}"
    
    local version=$(get_current_version)
    local git_hash=$(get_git_hash)
    
    echo "Title: NOZOMI: 3D White Matter Substrate Generation and Monte Carlo Diffusion Simulation"
    echo "Version: $version"
    echo "Description: A toolkit for generating realistic 3D axonal substrates and running Monte Carlo diffusion simulations for neuroimaging research."
    echo ""
    echo "Keywords: diffusion MRI, Monte Carlo simulation, white matter, axon modeling, neuroimaging"
    echo ""
    echo "Upload Type: Software"
    echo "License: MIT (or your chosen license)"
    echo ""
    echo "Related Identifiers:"
    echo "  - Repository: https://github.com/your-username/nozomi"
    echo "  - Git hash: $git_hash"
    echo ""
    echo "After Zenodo upload:"
    echo "1. Note the assigned DOI"
    echo "2. Update citations in documentation"
    echo "3. Create GitHub release linking to DOI"
}

show_citation() {
    echo -e "${BLUE}=== Citation Information ===${NC}"
    
    if command -v python3 &> /dev/null; then
        python3 -c "import sys; sys.path.insert(0, '.'); import nozomi; print(nozomi.get_citation_info())" 2>/dev/null || {
            echo "Could not import nozomi package. Using file-based information:"
            echo ""
            local version=$(get_current_version)
            local git_hash=$(get_git_hash)
            echo "NOZOMI $version"
            echo "Git hash: $git_hash"
            echo ""
            echo "To cite: NOZOMI Development Team. NOZOMI: 3D White Matter Substrate Generation and Monte Carlo Diffusion Simulation. Version $version. Git hash: ${git_hash:0:7}."
        }
    else
        echo "Python not available. Install Python to get full citation information."
    fi
}

# Main command processing
case "$1" in
    "current")
        show_current_version
        ;;
    "tag")
        shift
        VERSION=""
        MESSAGE=""
        DRY_RUN="false"
        
        while [[ $# -gt 0 ]]; do
            case $1 in
                -v|--version)
                    VERSION="$2"
                    shift 2
                    ;;
                -m|--message)
                    MESSAGE="$2"
                    shift 2
                    ;;
                --dry-run)
                    DRY_RUN="true"
                    shift
                    ;;
                *)
                    echo "Unknown option: $1"
                    print_usage
                    exit 1
                    ;;
            esac
        done
        
        create_version_tag "$VERSION" "$MESSAGE" "$DRY_RUN"
        ;;
    "prepare")
        prepare_release
        ;;
    "zenodo")
        generate_zenodo_info
        ;;
    "citation")
        show_citation
        ;;
    *)
        print_usage
        ;;
esac