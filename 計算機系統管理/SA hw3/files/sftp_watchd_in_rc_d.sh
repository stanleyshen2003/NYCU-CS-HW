#!/bin/sh
#
# $FreeBSD$
#

# netif is required for lo0 because syslogd tries to open a local socket
#
# PROVIDE: sftp_watchd
# REQUIRE:
# BEFORE:

. /etc/rc.subr

name="sftp_watchd"
rcvar="sftp_watchd_enable"
pidfile="/var/run/sftp_watchd.pid"
sftp_watchd_command="/usr/local/sbin/${name}"
command="/usr/sbin/daemon"
command_args="-P ${pidfile} -f -r ${sftp_watchd_command}"

load_rc_config $name
: ${sftp_watchd_enable:=no}

run_rc_command "$1"