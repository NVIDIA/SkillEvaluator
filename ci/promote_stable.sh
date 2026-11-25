#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

stable_ref=refs/tags/stable
version_pattern="^v(0|[1-9][0-9]*)[.](0|[1-9][0-9]*)[.](0|[1-9][0-9]*)$"
scanner=.github/workflows/skill-checks-stable.yml
publisher=.github/workflows/publish-skill-reports-stable.yml
gitlab=ci/gitlab/stable-tier1.yml
gitlab_template=ci/templates/stable/${gitlab##*/}
event=${GITHUB_EVENT_NAME:-}
expected_release=${STABLE_EXPECTED_RELEASE:-}
manual=false
mode=""
if (( $# )); then
    if [[ $# != 2 || ($1 != --manual && $1 != --verify-only) ]]; then
        echo "Usage: bash ci/promote_stable.sh {--manual|--verify-only} vX.Y.Z" >&2
        exit 1
    fi
    manual=true
    mode=$1
    expected_release=$2
fi

if [[ ($manual == true || $event == workflow_dispatch) && ! $expected_release =~ $version_pattern ]]; then
    echo "Manual promotion requires expected-release to be a canonical vX.Y.Z tag." >&2
    exit 1
fi

latest_release() {
    gh api repos/NVIDIA/SkillEvaluator/releases/latest --jq \
        'select(.immutable == true and .draft == false and .prerelease == false and .published_at != null)
         | .tag_name as $tag
         | ("skillevaluator-ci-" + ($tag | ltrimstr("v")) + ".zip") as $archive
         | (.assets[] | select(.name == "artifact-manifest.json")) as $manifest
         | (.assets[] | select(.name == $archive)) as $zip
         | [$tag, $manifest.digest, $manifest.size, $zip.digest, $zip.size] | @tsv'
}

latest=$(latest_release)
IFS=$'\t' read -r release manifest_digest manifest_size zip_digest zip_size <<< "$latest"
if [[ ! $release =~ $version_pattern || ! $manifest_digest =~ ^sha256:[a-f0-9]{64}$ ||
      ! $manifest_size =~ ^[0-9]{1,7}$ || ! $zip_digest =~ ^sha256:[a-f0-9]{64}$ ||
      ! $zip_size =~ ^[0-9]{1,8}$ || $latest == *$'\n'* ]]; then
    echo "The latest release must be a published, immutable stable version with an artifact manifest." >&2
    exit 1
fi
if [[ ($manual == true || $event == workflow_dispatch) && $release != "$expected_release" ]]; then
    echo "GitHub Latest is $release, but the expected release is $expected_release." >&2
    exit 1
fi

if [[ $manual == true ]]; then
    official_url=https://github.com/NVIDIA/SkillEvaluator
    fetch_url=$(git remote get-url --all origin)
    push_url=$(git remote get-url --push --all origin)
    if [[ ${fetch_url%.git} != "$official_url" || ${push_url%.git} != "$official_url" ]]; then
        echo "Manual promotion requires the official HTTPS SkillEvaluator origin for fetch and push." >&2
        exit 1
    fi
    if [[ -n $(git status --porcelain --untracked-files=all) ]]; then
        echo "Use a clean checkout of SkillEvaluator main for manual promotion." >&2
        exit 1
    fi
    git fetch --no-tags origin refs/heads/main
    main=$(git rev-parse 'FETCH_HEAD^{commit}')
    git fetch --no-tags origin "refs/tags/$release"
    release_commit=$(git rev-parse 'FETCH_HEAD^{commit}')
else
    main=$(git rev-parse refs/remotes/origin/main)
    release_commit=$(git rev-parse "refs/tags/$release^{commit}")
fi
git merge-base --is-ancestor HEAD "$main"
git merge-base --is-ancestor "$release_commit" "$main"

remote_refs=$(git ls-remote --refs origin refs/heads/stable "$stable_ref")
if [[ $(awk '$2 == "refs/heads/stable" {print $1}' <<< "$remote_refs") ]]; then
    echo "Remove the conflicting stable branch before promoting the channel." >&2
    exit 1
fi
previous=$(awk -v ref="$stable_ref" '$2 == ref {print $1}' <<< "$remote_refs")
previous_release=""
if [[ $previous ]]; then
    git fetch --no-tags origin "$stable_ref"
    previous_commit=$(git rev-parse "$previous^{commit}")
    previous_release=$(git show "$previous_commit:$scanner" | sed -n 's/^# SkillEvaluator release: //p')
    if [[ ! $previous_release =~ $version_pattern ]]; then
        echo "The existing stable tag does not contain a promoted SkillEvaluator release." >&2
        exit 1
    fi
    if [[ $(printf '%s\n' "$previous_release" "$release" | sort -V | tail -n 1) != "$release" ]]; then
        echo "Stable already serves a newer release than $release." >&2
        exit 1
    fi
fi

candidate=$release_commit
bootstrapping=false
if ! git cat-file -e "$release_commit:$scanner" 2>/dev/null ||
   ! git cat-file -e "$release_commit:$publisher" 2>/dev/null; then
    if [[ $release != v1.0.1 ]]; then
        echo "The published release must contain both stable workflows." >&2
        exit 1
    fi
    # The first published bundle predates the stable workflow entry points.
    if [[ $previous_release == "$release" ]]; then
        candidate=$previous_commit
    elif [[ $previous ]]; then
        echo "The v1.0.1 bootstrap requires that stable does not exist yet." >&2
        exit 1
    else
        bootstrapping=true
        candidate=${STABLE_BOOTSTRAP_SHA:-}
        if [[ ($manual != true && $event != workflow_dispatch) || ! $candidate =~ ^[0-9a-f]{40}$ ||
              $(git cat-file -t "$candidate" 2>/dev/null) != commit ]]; then
            echo "Bootstrap v1.0.1 with the full reviewed main SHA as STABLE_BOOTSTRAP_SHA locally or bootstrap-sha in Actions." >&2
            exit 1
        fi
    fi
fi
git merge-base --is-ancestor "$candidate" "$main"
git merge-base --is-ancestor "$release_commit" "$candidate"
version=${release#v}
bundle="ci/bundles/$version"
archive="skillevaluator-ci-$version.zip"
git diff --exit-code "$release_commit" "$candidate" -- "$bundle"

# Run the validator from main and read the chosen commit's files as data.
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
assets="$scratch/release-assets"
mkdir -p "$assets" "$scratch/$bundle" "$scratch/.github/workflows" "$scratch/ci/templates/stable" "$scratch/ci/gitlab"
for file in release.json verification-receipt.json "$archive"; do
    git cat-file blob "$release_commit:$bundle/$file" > "$scratch/$bundle/$file"
done
for workflow in "$scanner" "$publisher"; do
    template="ci/templates/stable/${workflow##*/}"
    git cat-file blob "$candidate:$template" > "$scratch/$template"
    git cat-file blob "$candidate:$workflow" > "$scratch/$workflow"
done
for file in "$gitlab_template" "$gitlab"; do
    if git cat-file -e "$candidate:$file" 2>/dev/null; then
        git cat-file blob "$candidate:$file" > "$scratch/$file"
    fi
done
if [[ ! -f "$scratch/$gitlab" ]]; then
    if [[ $bootstrapping == true ]]; then
        echo "The reviewed bootstrap commit must contain the GitLab stable include." >&2
        exit 1
    fi
    if [[ $previous ]] && git cat-file -e "$previous_commit:$gitlab" 2>/dev/null; then
        echo "The next stable release must retain the GitLab stable include." >&2
        exit 1
    fi
fi
gh release download "$release" --repo NVIDIA/SkillEvaluator --pattern artifact-manifest.json --dir "$assets"
python3 ci/stable_channel.py "$scratch/$bundle" --check \
    --templates "$scratch/ci/templates/stable" --output "$scratch/.github/workflows" \
    --gitlab-output "$scratch/$gitlab" \
    --publication "$assets/artifact-manifest.json" "$release_commit" "$manifest_digest" "$manifest_size" \
    "$zip_digest" "$zip_size"

if [[ $previous_release == "$release" ]]; then
    if [[ $previous_commit != "$candidate" ]]; then
        echo "The existing stable tag for $release points to a different commit." >&2
        exit 1
    fi
    if [[ $mode != --verify-only ]]; then
        echo "Stable already serves $release."
        exit 0
    fi
fi

# Keep the checked manifest and CI archive as the reference for both public indexes.
cp "$scratch/$bundle/$archive" "$assets/"
gh release download "$release" --repo NVIDIA/SkillEvaluator \
    --pattern "skillevaluator-$version-py3-none-any.whl" --pattern "skillevaluator-$version.tar.gz" \
    --pattern SHA256SUMS.txt --dir "$assets"
python3 ci/product_release.py verify-indexes --version "$version" --assets "$assets"

if [[ $(latest_release) != "$latest" ]]; then
    echo "The latest release changed during promotion. Retry from the release workflow." >&2
    exit 1
fi
push=(git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push --no-follow-tags
    "--force-with-lease=$stable_ref:$previous" origin "$candidate:$stable_ref")
if [[ $manual != true ]]; then
    GH_TOKEN="${STABLE_PUSH_TOKEN:?A stable channel writer token is required}" "${push[@]}"
elif [[ $mode == --verify-only ]]; then
    echo "Verified $release at $candidate for stable promotion."
    exit 0
else
    "${push[@]}"
fi
echo "Stable now serves $release from $candidate."
