# follow the step in https://www.alibabacloud.com/blog/how-to-build-nginx-from-source-on-ubuntu-20-04-lts_597793, but install a version higher then 1.25.0 to have support for http3
# I forgot my ./configure ... , the main thing is to add -with-http_v3_module

# update /etc/hosts
# update /etc/nginx/nginx.conf
su -
mkdir -p /home/judge/www/137.cs.nycu
mkdir -p /home/judge/www/10.113.137.11
touch /home/judge/www/137.cs.nycu/index.html
touch /home/judge/www/10.113.137.11/index.html
echo "2023-nycu.sa-hw4-vhost" > /home/judge/www/137.cs.nycu/index.html
echo "2023-nycu.sa-hw4-ip" > /home/judge/www/10.113.137.11/index.html
chmod -R 775 /home/judge/www

# install certbot to get the certificate from ca server
# rename the .pem given by the TAs to .crt


sudo apt-get install certbot
sudo certbot certonly --standalone -d 137.cs.nycu --server https://ca.nasa.nycu:9000/acme/acme/directory


sudo apt-get install apache2-utils
sudo htpasswd -c /home/judge/www/.htpasswd sa-admin



# refernce https://curl.se/docs/http3.html
git clone --depth 1 -b openssl-3.1.4+quic https://github.com/quictls/openssl
cd openssl
./config enable-tls1_3 --prefix=/usr/local/etc
make
sudo make install
# Build nghttp3
cd ..
git clone -b v1.1.0 https://github.com/ngtcp2/nghttp3
cd nghttp3
autoreconf -fi
./configure --prefix=/usr/local/etc --enable-lib-only
make
make install

# Build ngtcp2
cd ..
git clone -b v1.1.0 https://github.com/ngtcp2/ngtcp2
cd ngtcp2
autoreconf -fi
./configure PKG_CONFIG_PATH=/usr/local/etc/lib64/pkgconfig:/usr/local/etc/lib/pkgconfig LDFLAGS="-Wl,-rpath,/usr/local/etc/lib64" --prefix=/usr/local/etc --enable-lib-only
make
make install

# Build nghttp2
sudo apt-get install g++ clang make binutils autoconf automake \
  autotools-dev libtool pkg-config \
  zlib1g-dev libcunit1-dev libssl-dev libxml2-dev libev-dev \
  libevent-dev libjansson-dev \
  libc-ares-dev libjemalloc-dev libsystemd-dev \
  ruby-dev bison libelf-dev
git clone https://github.com/nghttp2/nghttp2.git
git submodule update --init
autoreconf -i
./configure PKG_CONFIG_PATH=/usr/local/etc/lib64 --prefix=/usr/local/etc --enable-lib-only
make 
sudo make install

# Build curl
cd ..
git clone https://github.com/curl/curl
cd curl
autoreconf -fi
LDFLAGS="-Wl,-rpath,/usr/local/etc/lib64" ./configure --with-openssl=/usr/local/etc --with-nghttp3=/usr/local/etc --with-ngtcp2=/usr/local/etc --with-nghttp2=/usr/local/etc
make
make install


# firewall
sudo iptables -A INPUT -p icmp --icmp-type any -s 10.113.137.254 -j ACCEPT
sudo iptables -A INPUT -p icmp --icmp-type any -j DROP
sudo iptables -A INPUT -s 10.113.137.0/24 -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -s 10.113.137.0/24 -p tcp --dport 443 -j ACCEPT
sudo iptables -A INPUT -s 10.113.137.0/24 -p tcp --dport 3443 -j ACCEPT
sudo iptables -A INPUT -s 10.113.137.0/24 -p udp --dport 3443 -j ACCEPT
sudo iptables -A INPUT -p udp --dport 3443 -j DROP
sudo iptables -A INPUT -p tcp --dport 443 -j DROP
sudo iptables -A INPUT -p tcp --dport 3443 -j DROP
sudo iptables -A INPUT -p tcp --dport 80 -j DROP

# setup jail
sudo apt install fail2ban
cp /etc/fail2ban/fail2ban.conf /etc/fail2ban/fail2ban.local
# modify fail2ban.local