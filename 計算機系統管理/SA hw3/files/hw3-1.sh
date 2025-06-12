#!/bin/sh
mkdir -p /home/sftp/public /home/sftp/hidden
mkdir -p /home/sftp/hidden/treasure /home/sftp/hidden/.exe
touch /home/sftp/hidden/treasure/secret

chown -R sysadm:sftp /home/sftp/public /home/sftp/hidden
chmod -R 775 /home/sftp/public
chmod 2771 /home/sftp/hidden
chmod +t /home/sftp/public
chmod 777 /home/sftp/hidden/.exe
chmod -R 777 /home/sftp/hidden/treasure

cp -r /usr/home/judge/.ssh /usr/home/sftp/.ssh