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

IMAGE ?= ghcr.io/gabintb/embedx
VERSION := $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)

image:
	docker build -t $(IMAGE):$(VERSION) -t $(IMAGE):latest .

# The failure this catches is the one that is otherwise silent: an image with
# no `serve` still builds, still starts, and still passes the healthcheck.
# Needs no GPU -- `--help` short-circuits before any device is touched.
image-verify:
	@echo "verifying $(IMAGE):$(VERSION)"
	@docker run --rm --entrypoint embedx $(IMAGE):$(VERSION) serve --help > /dev/null \
	  || { echo "FATAL: no 'serve' in $(IMAGE):$(VERSION) -- built without the HTTP layer"; exit 1; }
	@docker run --rm --entrypoint embedx $(IMAGE):$(VERSION) --version
	@echo "ok: serve present"

# Requires `docker login ghcr.io -u <you>` with a PAT carrying write:packages.
# Pushes the version tag and moves `latest`; run it only for a real release.
image-push: image image-verify
	docker push $(IMAGE):$(VERSION)
	docker push $(IMAGE):latest
	@echo "pushed $(IMAGE):$(VERSION) and :latest"
