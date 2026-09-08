#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${1:-/mnt/ML_projects/conda_envs/dante_env}"
LIBDIR="${ENV_PREFIX%/}/lib"

if [[ ! -d "${LIBDIR}" ]]; then
  echo "ERROR: lib dir not found: ${LIBDIR}" >&2
  exit 1
fi

fix_link() {
  local link_name="$1"
  local target_name="$2"
  local link_path="${LIBDIR}/${link_name}"
  local target_path="${LIBDIR}/${target_name}"

  if [[ ! -e "${target_path}" ]]; then
    echo "SKIP: target missing: ${target_path}"
    return 0
  fi

  if [[ -L "${link_path}" ]]; then
    ln -sfn "${target_name}" "${link_path}"
    echo "OK: ${link_name} -> ${target_name} (was symlink)"
    return 0
  fi

  if [[ -e "${link_path}" ]]; then
    local sz
    sz="$(wc -c < "${link_path}" | tr -d ' ')"
    if [[ "${sz}" -lt 4096 ]]; then
      mv -f "${link_path}" "${link_path}.bak.$(date +%Y%m%d_%H%M%S)"
      ln -s "${target_name}" "${link_path}"
      echo "OK: ${link_name} -> ${target_name} (replaced tiny file ${sz}B)"
      return 0
    fi
  fi

  # not present or already a normal-sized file: still ensure correct symlink exists
  if [[ ! -e "${link_path}" ]]; then
    ln -s "${target_name}" "${link_path}"
    echo "OK: ${link_name} -> ${target_name} (created)"
  else
    echo "SKIP: ${link_path} exists and looks normal"
  fi
}

# These were observed broken in dante_env (file too short). Keep list explicit to avoid accidental changes.
fix_link "libz.so.1" "libz.so.1.2.13"
fix_link "libffi.so.8" "libffi.so.8.1.2"
fix_link "libffi.so" "libffi.so.8.1.2"
fix_link "libffi.so.7" "libffi.so.8.1.2"
fix_link "libstdc++.so.6" "libstdc++.so.6.0.29"
fix_link "libstdc++.so" "libstdc++.so.6.0.29"
fix_link "libgfortran.so.5" "libgfortran.so.5.0.0"
fix_link "libgfortran.so" "libgfortran.so.5.0.0"
fix_link "libquadmath.so.0" "libquadmath.so.0.0.0"
fix_link "libquadmath.so" "libquadmath.so.0.0.0"
fix_link "libgomp.so.1" "libgomp.so.1.0.0"
fix_link "libgomp.so" "libgomp.so.1.0.0"

