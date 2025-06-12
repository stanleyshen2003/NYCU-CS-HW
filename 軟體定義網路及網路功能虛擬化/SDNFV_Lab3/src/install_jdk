#!/bin/bash

readonly CSI_RESET='\033[0m'
readonly CSI_BOLD_GREEN='\033[1;32m'

info() {
  echo -e "${CSI_BOLD_GREEN}$1${CSI_RESET}"
}

install_dependencies() {
  sudo apt-get -q update
  sudo apt-get -yq install gnupg curl 
}

add_azul_public_key() {
  sudo apt-key adv \
    --keyserver hkp://keyserver.ubuntu.com:80 \
    --recv-keys 0xB1998361219BD9C9
}

add_azul_apt_repo() {
  local package_name='zulu-repo_1.0.0-3_all.deb'

  curl -O "https://cdn.azul.com/zulu/bin/${package_name}"
  sudo apt-get install "./${package_name}"
  sudo apt-get update
  rm "./${package_name}"
}

install_zulu_jdk_11() {
  sudo apt-get install zulu11-jdk
}

main() {
  info 'Start to install Azul Zulu JDK 11.'

  set -e

  install_dependencies
  add_azul_public_key
  add_azul_apt_repo
  install_zulu_jdk_11

  info '***********************************************************'
  info '* Installation of Azul Zulu JDK 11 finished successfully! *'
  info '***********************************************************'
}

main "$@"
