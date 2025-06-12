#!/bin/bash

readonly BAZELISK_VERSION='1.12.0'

readonly ONOS_DIR="${HOME}/onos"
readonly ONOS_VERSION='2.7.0'

readonly MININET_DIR="${HOME}/mininet"
readonly MININET_VERSION='2.3.0'

readonly OVS_DIR="${HOME}/ovs"
readonly OVS_VERSION='2.17.2'
readonly OVS_SERVICE_NAME='ovs'

readonly CSI_RESET='\033[0m'
readonly CSI_BOLD_GREEN='\033[1;32m'

info() {
  echo -e "${CSI_BOLD_GREEN}$1${CSI_RESET}"
}

install_bazelisk() {
  if [[ -n "$(which bazel)" ]]; then
    info 'Skip the installation of Bazelisk.'
    return 0
  fi

  info "Start installing Bazelisk v${BAZELISK_VERSION}..."

  cd ~
  install_bazelisk_deps
  install_bazelisk_core

  info 'Finished the installation of Bazelisk.'
}

install_bazelisk_deps() {
  info 'Start installing Bazelisk dependencies...'

  sudo apt update
  sudo apt install -y wget
}

install_bazelisk_core() {
  info 'Start installing Bazelisk core...'

  wget "https://github.com/bazelbuild/bazelisk/releases/download/v${BAZELISK_VERSION}/bazelisk-linux-amd64"
  sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
  sudo chmod +x /usr/local/bin/bazel
}

install_onos() {
  if grep -Fq "${ONOS_DIR}" ~/.bashrc; then
    info 'Skip the installation of ONOS.'
    return 0
  fi

  info "Start installing ONOS v${ONOS_VERSION}..."

  cd ~
  install_onos_deps
  ensure_onos_source
  build_onos
  add_ssh_rsa_for_localhost
  add_onos_dev_env

  info 'Finished the installation of ONOS.'
}

install_onos_deps() {
  info 'Start installing ONOS dependencies...'

  sudo apt update
  sudo apt install -y \
    build-essential \
    bzip2 \
    curl \
    git \
    maven \
    python2 \
    python3 \
    unzip \
    zip
  sudo ln -sf python2 /usr/bin/python
}

ensure_onos_source() {
  if [[ ! -d "${ONOS_DIR}" ]]; then
    info 'Start cloning ONOS source code...'
    git clone https://gerrit.onosproject.org/onos "${ONOS_DIR}"
  fi

  cd "${ONOS_DIR}"
  git checkout "${ONOS_VERSION}"
}

build_onos() {
  info 'Start building ONOS...'

  cd "${ONOS_DIR}"
  bazel build onos
}

add_ssh_rsa_for_localhost() {
  info 'Adding ssh-rsa algorithm for localhost...'

  if [[ ! -f "${HOME}/.ssh/config" ]]; then
    mkdir -p ~/.ssh
    touch ~/.ssh/config
  fi

  echo 'Host localhost
  HostkeyAlgorithms +ssh-rsa
  PubkeyAcceptedAlgorithms +ssh-rsa' >> ~/.ssh/config
}

add_onos_dev_env() {
  info 'Adding ONOS developer environment...'

  echo "export ONOS_ROOT=${ONOS_DIR}
. \${ONOS_ROOT}/tools/dev/bash_profile" >> ~/.bashrc
}

install_mininet() {
  if [[ -n "$(which mn)" ]]; then
    info 'Skip the installation of Mininet.'
    return 0
  fi

  info "Start installing Mininet v${MININET_VERSION}..."

  cd ~
  install_mininet_deps
  ensure_mininet_source
  install_mininet_core

  info 'Finished the installation of Mininet.'
}

install_mininet_deps() {
  info 'Start installing Mininet dependencies...'

  sudo apt update
  sudo apt install -y \
    arping \
    git \
    iputils-ping
}

ensure_mininet_source() {
  if [[ ! -d "${MININET_DIR}" ]]; then
    info 'Start cloning Mininet source code...'
    git clone https://github.com/mininet/mininet.git "${MININET_DIR}"
  fi

  cd "${MININET_DIR}"
  git checkout "${MININET_VERSION}"
}

install_mininet_core() {
  info 'Start installing Mininet core...'

  # Using -V to install OVS will result in some missing package errors.
  # Therefore, we install OVS through a released tarball instead.
  PYTHON=python3 "${MININET_DIR}/util/install.sh" -n
}

install_ovs() {
  if systemctl cat "${OVS_SERVICE_NAME}" > /dev/null 2>&1; then
    info 'Skip the installation of Open vSwitch.'
    return 0
  fi

  info "Start installing Open vSwitch v${OVS_VERSION}..."

  cd ~
  install_ovs_deps
  ensure_ovs_release
  install_ovs_core
  register_ovs_service

  info 'Finished the installation of Open vSwitch.'
}

install_ovs_deps() {
  info 'Start installing Open vSwitch dependencies...'

  sudo apt update
  sudo apt install -y \
    build-essential \
    python3 \
    wget
}

ensure_ovs_release() {
  if [[ -d "${OVS_DIR}" ]]; then
    return 0
  fi

  info 'Start downloading Open vSwitch release...'

  local dir_name="openvswitch-${OVS_VERSION}"
  local tarball_name="${dir_name}.tar.gz"

  wget "https://www.openvswitch.org/releases/${tarball_name}"
  tar -xf "${tarball_name}"
  mv "${dir_name}" "${OVS_DIR}"
}

install_ovs_core() {
  info 'Start installing Open vSwitch core...'

  cd "${OVS_DIR}"
  ./configure
  make
  sudo make install
}

register_ovs_service() {
  info 'Start registering Open vSwitch service...'

  local filename="${OVS_SERVICE_NAME}.service"
  local ovsctl='/usr/local/share/openvswitch/scripts/ovs-ctl'

  touch "${filename}"
  echo "[Unit]
Description=Open vSwitch v${OVS_VERSION} service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=${ovsctl} start
ExecStop=${ovsctl} stop

[Install]
WantedBy=multi-user.target" >> "${filename}"
  sudo mv "${filename}" "/etc/systemd/system/${filename}"

  sudo systemctl enable "${OVS_SERVICE_NAME}"
  sudo systemctl start "${OVS_SERVICE_NAME}"
}

main() {
  info 'Start to set up SDN environment.'

  set -e

  install_bazelisk
  install_onos
  install_mininet
  install_ovs

  info '************************************************'
  info '* The environment setup finished successfully! *'
  info '************************************************'
}

main "$@"
