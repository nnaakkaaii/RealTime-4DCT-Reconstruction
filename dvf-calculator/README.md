# DVF Calculator

`DVF Calculator` is a Dockerized tool developed to compute Displacement Vector Fields (DVF) from 4D-CT data sets. Utilizing the robust image registration capabilities of SimpleITK, this tool simplifies the process of transforming sequential CT scans into their associated DVF format.

## Prerequisites

- Docker installed on your machine. If not yet installed, obtain it from [Docker's official site](https://docs.docker.com/get-docker/).

## Directory Structure

```
dvf-calculator/
│
├── Dockerfile
├── README.md
├── requirements.txt
└── script.py
```

## Building the Docker Image

To build the Docker image, navigate to the `dvf-calculator` and execute:

```bash
docker build -t dvf-calculator .
```

This will compile the Docker image and tag it as `dvf-calculator`.

## Running the Tool

Once you've built the image, you can utilize the DVF Calculator for any directory containing 4D-CT `.npz` files by executing:

```bash
docker run --rm -v $(pwd)/../data/4D-Lung/npz1:/app/data dvf-calculator python script.py /app/data --shrink-factors 16 8 4 2 1 --smoothing-sigmas 5 4 2 1 0 --sampling-percentage 0.5
```

Ensure you replace `/path/to/data` with the actual path to your data directory. The calculated DVF outputs will be stored within the same directory with the `_dvf.npz` suffix.

## Contributions

We welcome and appreciate pull requests. For significant alterations or features, kindly open an issue beforehand to deliberate on your proposed changes.

## License

This project is licensed under the MIT License. For detailed information, please refer to the [LICENSE.md](LICENSE.md) file.
