## Releasing

The release artifacts for tf-encrypted are built entirely through Circle CI
(our continuous integration tool). It detects when a new tag is pushed to
GitHub and then runs the build. If it passes, then the `deploy` job will run
which will trigger the `make push` rule in our Makefile which will subseuqently
build and push all release artifacts.

Today, the only release artifact produced from this repository is a
[Docker](https://www.docker.com) container.

If a release candidate (e.g. `X.Y.Z-rc.#`) is pushed then Circle CI will build
the container (`mortendahl/tf-encrypted:X.Y.Z-rc.#`) and push it to Docker Hub
*without* updating the `mortendahl/tf-encrypted:latest` tag.

If a release tag (e.g. `X.Y.Z`) is pushed then Circle CI will build the
containers and push both `mortendahl/tf-encrypted:X.Y.Z` and
`mortendahl/tf-encrypted:lateset` to Docker Hub.

### Whats the flow?

Anyone with write access can trigger a release candidate or release to be
created by pushing a tag to GitHub! We've documented the flow for releasing
tf-encrypted below:

1. Create a Release Candidate by creating a tag on master in the form of
   `X.Y.Z-rc.#`. If this is the first release candidate in the series for this
   tag the number should begin at 0 (e.g. `git tag 0.1.0-rc.0`).
2. Push the tag up to GitHub which will trigger a build in Circle CI (e.g. `git
   push origin 0.1.0-rc.0`). Once this build is complete, Circle CI will build
   the release artifacts to their respective repositories (Docker Hub, etc.).
3. Test the Release Artifact to make sure it passes our bar of quality. If any
   bugs are found or other quality issues create a new release candidate by
   proceeding back to the first step! Don't forget to increment the Release
   Candidate version!
4. Once we're satisfied with the quality of the release candidate, it's time to
   cut the actual release! This is done once again by creating a tag without
   the `-rc.#` (e.g. `git tag 0.1.0`) and push it again (e.g. `git push origin
   0.1.0`). Once the build is done, the release is available in the wild!

Have a question about the process or have a suggestion on how to improve it?
Don't hesitate to open an [issue](https://github.com/mortendahl/tf-encrypted/issues/new)
with your thoughts!

### Whats with the tags?

Development follows [`semver v2.0`](https://semver.org/) versioning
specification. If the major has changed (e.g. the X in `X.Y.Z`) then a breaking
API change has occurred requiring any dependent users to make changes to their
code. If the minor (e.g. the Y in `X.Y.Z`) has changed then one or more new
features have been added. While if the patch has changed (e.g. the Z in
`X.Y.Z`) then only a fix has been implemented.

This makes it easy for developers of tf-encrypted and users alike to communicate
about the risk involved in uptaking a new version of tf-encrypted.
