#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

stable_ref=refs/tags/stable
version_pattern="^v(0|[1-9][0-9]*)[.](0|[1-9][0-9]*)[.](0|[1-9][0-9]*)$"
scanner=.github/workflows/skill-checks-stable.yml
publisher=.github/workflows/publish-skill-reports-stable.yml

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

main=$(git rev-parse refs/remotes/origin/main)
release_commit=$(git rev-parse "refs/tags/$release^{commit}")
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
        candidate=${STABLE_BOOTSTRAP_SHA:-}
        if [[ ${GITHUB_EVENT_NAME:-} != workflow_dispatch || ! $candidate =~ ^[0-9a-f]{40}$ ||
              $(git cat-file -t "$candidate" 2>/dev/null) != commit ]]; then
            echo "Bootstrap v1.0.1 by dispatching this workflow with a reviewed main commit as bootstrap-sha." >&2
            exit 1
        fi
    fi
fi
git merge-base --is-ancestor "$candidate" "$main"
git merge-base --is-ancestor "$release_commit" "$candidate"
bundle="ci/bundles/${release#v}"
git diff --exit-code "$release_commit" "$candidate" -- "$bundle"

# Run the validator from main and read the chosen commit's files as data.
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/$bundle" "$scratch/.github/workflows" "$scratch/ci/templates/stable"
for file in release.json verification-receipt.json "skillevaluator-ci-${release#v}.zip"; do
    git cat-file blob "$release_commit:$bundle/$file" > "$scratch/$bundle/$file"
done
for workflow in "$scanner" "$publisher"; do
    template="ci/templates/stable/${workflow##*/}"
    git cat-file blob "$candidate:$template" > "$scratch/$template"
    git cat-file blob "$candidate:$workflow" > "$scratch/$workflow"
done
gh release download "$release" --repo NVIDIA/SkillEvaluator --pattern artifact-manifest.json --dir "$scratch"
python3 ci/stable_channel.py "$scratch/$bundle" --check \
    --templates "$scratch/ci/templates/stable" --output "$scratch/.github/workflows" \
    --publication "$scratch/artifact-manifest.json" "$release_commit" "$manifest_digest" "$manifest_size" \
    "$zip_digest" "$zip_size"

if [[ $previous_release == "$release" ]]; then
    if [[ $previous_commit != "$candidate" ]]; then
        echo "The existing stable tag for $release points to a different commit." >&2
        exit 1
    fi
    echo "Stable already serves $release."
    exit 0
fi

if [[ $(latest_release) != "$latest" ]]; then
    echo "The latest release changed during promotion. Retry from the release workflow." >&2
    exit 1
fi
GH_TOKEN="${STABLE_PUSH_TOKEN:?A stable channel writer token is required}" \
    git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push \
    --force-with-lease="$stable_ref:$previous" origin "$candidate:$stable_ref"
echo "Stable now serves $release from $candidate."
