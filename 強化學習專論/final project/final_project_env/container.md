# Usage

The following is the example of how to use the container via podman.
You can also use docker.

### Build an image

```bash
podman build -t competition .
```

### Run a container

```bash
# Choose your container name
podman run -it --name comp_container --network host -v "$(pwd):/workingdirectory" competition
```

```bash
podman exec -it comp_container /bin/bash
```

### Run evaluation

```bash
# Choose the correct student ID and pick up a port number
python3 server.py --sid 0716092 --port 12345 --scenario ...
```


### Remove the container

```bash
podman rm -f comp_container
```