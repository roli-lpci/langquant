#!/usr/bin/env bash
# verify.sh — thin wrapper over `hermes-seal verify .`. Exit code is the answer.
set -euo pipefail
exec hermes-seal verify "${1:-.}" "${@:2}"
