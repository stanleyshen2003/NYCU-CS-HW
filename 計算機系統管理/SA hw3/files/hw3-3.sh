#!/bin/bash
sudo gpart create -s gpt /dev/ada1
sudo gpart add -t freebsd-zfs -l mypool-1 -s 250M /dev/ada1
sudo gpart add -t freebsd-zfs -l mypool-2 -s 250M /dev/ada1
sudo gpart add -t freebsd-zfs -l mypool-3 -s 250M /dev/ada1
sudo gpart add -t freebsd-zfs -l mypool-4 -s 250M /dev/ada1

# Create a RAID10 pool 
sudo zpool create mypool mirror /dev/gpt/mypool-1 /dev/gpt/mypool-2 mirror /dev/gpt/mypool-3 /dev/gpt/mypool-4

sudo zfs set mountpoint=/usr/home/sftp mypool

sudo zfs set compression=lz4 mypool
sudo zfs set atime=off mypool
sudo zfs create mypool/public
sudo zfs create mypool/hidden




# note
# dmesg -> look storage.
# lsblk -> look pages.
# sudo mount /dev/ada1 [path] -> mount disk
# sudo zfs mount mypool -> mount mypool's path
# /boot/loader.conf -> enable gtpid = "1"