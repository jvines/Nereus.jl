# Merge message templates

These exist for one reason: without them Forgejo appends

    Reviewed-on: https://<forge-host>/<owner>/<repo>/pulls/<n>

to every merge commit it creates from the web UI, and this repository is
push-mirrored to a public host on every push. That trailer publishes the
internal forge hostname, in the commit MESSAGE, where neither gate catches it:
`tools/ci/pre-push` only sees commits pushed from a clone, and the merge commit
is created server-side; CI scans the working tree, not messages.

Two such trailers reached the public mirror on 2026-09-25 before this was
noticed. The templates override the default so the message carries the title,
the PR number and the branches -- and nothing that names the forge.

Forgejo reads these from `.gitea/` for Gitea compatibility, and only from the
repository's DEFAULT branch (`main`), whatever the PR's base branch is. A copy
on `develop` alone is ignored: the merge of a PR into `develop` leaked the
trailer with templates sitting on `develop` and none on `main`. The available
placeholders are ${PullRequestTitle}, ${PullRequestIndex}, ${HeadBranch},
${BaseBranch}, ${PullRequestPosterName} and ${PullRequestDescription}; none of
them expand to a URL, which is the point.

If a merge commit ever shows a `Reviewed-on:` line again, this override has
stopped being honoured -- check it before merging anything else.
