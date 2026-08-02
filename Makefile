# Local mirror of the CI gate (.github/workflows/ci.yml), plus the container
# image release path.

.PHONY: lint fmt test gate image image-verify image-push

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src/embedx

fmt:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest -m "not gpu"

gate: lint test

# --------------------------------------------------------------------------- #
# Container image — built and pushed from a GPU host, deliberately not in CI.
#
# The image is ~12.3 GB. A GitHub-hosted runner starts with roughly 21 GB free
# and this build needs 20-25 GB, because the 7.36 GB /opt/venv layer is
# materialised in both the builder and the runtime stage; it would need a
# disk-cleanup step to fit at all. More to the point, a hosted runner has no
# GPU, so it could not run the image it published. Building here means the
# artifact is pushed from the machine that can actually verify it.
#
# `EMBEDX_BUILD_SERVER=1` is NOT passed here and must not be: it lives inside
# the Dockerfile's install step, so no caller can forget it. tests/test_docker.py
# asserts that.
# --------------------------------------------------------------------------- #

# ONE build, ONE verification, two registries. The tags below all name the
# same image id, so pushing to both distributes bytes that were checked here
# rather than two independently-produced images that could differ.
IMAGE_REPO ?= gabintb/embedx
REGISTRIES ?= docker.io/$(IMAGE_REPO) ghcr.io/$(IMAGE_REPO)
VERSION := $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)

# -t <registry>:<version> -t <registry>:latest, for each registry.
IMAGE_TAGS := $(foreach reg,$(REGISTRIES),-t $(reg):$(VERSION) -t $(reg):latest)

image:
	docker build $(IMAGE_TAGS) .

# The failure this catches is the one that is otherwise silent: an image with
# no `serve` still builds, still starts, and still passes the healthcheck.
# Needs no GPU -- `--help` short-circuits before any device is touched.
#
# Run per registry tag, not once against the image id. The tags do resolve to
# the same bytes, so this is not re-verifying the build -- it is verifying
# that each tag ACTUALLY POINTS AT the thing that was verified. A typo in
# REGISTRIES, or a stale tag left by an earlier build, fails here instead of
# being pushed. Adding a second destination must not make the gate weaker.
image-verify:
	@for reg in $(REGISTRIES); do \
	  echo "verifying $$reg:$(VERSION)"; \
	  docker run --rm --entrypoint embedx $$reg:$(VERSION) serve --help > /dev/null \
	    || { echo "FATAL: no 'serve' in $$reg:$(VERSION) -- built without the HTTP layer"; exit 1; }; \
	  docker run --rm --entrypoint embedx $$reg:$(VERSION) --version > /dev/null \
	    || { echo "FATAL: $$reg:$(VERSION) does not run"; exit 1; }; \
	done
	@echo "ok: serve present in every tagged registry ($(VERSION))"

# Credentials come from the local docker credential store and are NEVER in
# this file or the repository. Log in once per registry, each with an access
# token rather than a password:
#
#   docker login docker.io   -u gabintb   # Docker Hub access token
#   docker login ghcr.io     -u GabinTB   # GitHub PAT with write:packages
#
# Pushes the version tag and moves `latest`; run it only for a real release.
# The build and the verification are prerequisites, so `latest` cannot move
# to an image that has not passed the gate.
image-push: image image-verify
	@for reg in $(REGISTRIES); do \
	  docker push $$reg:$(VERSION) || exit 1; \
	  docker push $$reg:latest || exit 1; \
	done
	@echo "pushed $(VERSION) and latest to: $(REGISTRIES)"
