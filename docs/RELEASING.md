## Releasing

The release artifacts for tf-encrypted are built entirely through Circle CI
(our continuous integration tool). It detects when a new tag is pushed to
GitHub and then runs the build. If it passes, then the `deploy` job will run
which will trigger the `make push` rule in our Makefile which will subsequently
build and push all release artifacts.

Today, the following artifacts are produced from this repository:

- a docker container which is available on docker hub as [tfencrypted/tf-encrypted](https://hub.docker.com/r/tfencrypted/tf-encrypted))
- a python package registered on pypi as [tf-encrypted](https://pypi.org/project/tf-encrypted)

If a release candidate tag (e.g. `X.Y.Z-rc#`) is pushed then Circle CI will
build these artifacts and push them to the their respective repositories. A
release candidate *will not* update the `latest` tag on docker hub.

If a release tag (e.g. `X.Y.Z`) is pushed then Circle CI will build and push
all artifacts along with updating the `latest` tag on docker hub.

### Whats the flow?

Anyone with write access can trigger a release candidate or release to be
created by pushing a tag to GitHub! We've documented the flow for releasing
tf-encrypted below:

1. Create a release candidate branch based off master (e.g.
   `release-0.1.0-rc0`) and update the `setup.py`, `CHANGELOG.md`, `meta.yaml`,
   and `docs/source/conf.py` and attempt to merge it into master. If this
   is the first release candidate in the series for this version the
   number should begin at 0 (e.g. `0.1.0-rc0`).
2. Create a tag off of the commit that merges the release candidate branch into
   master for the given version in setup.py (e.g. `git tag 0.1.0-rc0`) and push
   to GitHub (e.g. `git push origin 0.1.0-rc0`). This will trigger a deploy on
   Circle CI which will push the artifacts to PyPI and Docker Hub.
3. Once the build on Circle CI has passed create pull down the library from
   pypi and ensure the different examples in our library and documentation work
   as expected. If at any point a bug is found repeat steps 1-3 until the
   release works as expected.
4. Once the release candidate work as expected, create a new branch based off
   master (e.g. `release-0.1.0`), update the files above with the true release
   version, and merge it into the master branch on TF Encrypted's GitHub. Once
   you have verified that the CircleCI jobs have finished as we did for the
   release candidate, create a tag (e.g. `git tag 0.1.0`) off of the
   merge commit and push it to github (e.g. `git push origin 0.1.0`).
5. Once the build on Circle CI has passed for our full release it should be
   available to the world to start using! Don't forget to tweet about it :)

**NOTE**: You must update *and* commit to master a new version of `setup.py`
everytime you want to tag a new version of `tf-encrypted`. Make sure you push
the changes to `master` of [tf-encrypted on github](https://github.com/tf-encrypted/tf-encrypted).

Have a question about the process or have a suggestion on how to improve it?
Don't hesitate to open an [issue](https://github.com/tf-encrypted/tf-encrypted/issues/new)
with your thoughts!

### Whats with the tags?

This project follows the [PEP 440](https://www.python.org/dev/peps/pep-0440/)
versioning specification. If the major has changed (e.g. the X in `X.Y.Z`) then
a breaking API change has occurred requiring any dependent users to make
changes to their code. If the minor (e.g. the Y in `X.Y.Z`) has changed then
one or more new features have been added. While if the patch has changed (e.g.
the Z in `X.Y.Z`) then only a fix has been implemented.

This makes it easy for developers of tf-encrypted and users alike to communicate
about the risk involved in uptaking a new version of tf-encrypted.

### What's with the CHANGELOG?

Any time we make a change that could impact a consumer of tf-encrypted whether
it be a bug fix, api improvement, performance enhancement, or resolve a
security issue a note is added to the CHANGELOG.md under the `[Unreleased]`
section as a part of the pull request changing the code. This makes it easy for
any consumers of the project to understand *exactly* what has changed in any
given version.

When a release is created the `[Unreleased]` block is updated to the release
version and a new `[Unreleased]` block is put at the top of the file.
