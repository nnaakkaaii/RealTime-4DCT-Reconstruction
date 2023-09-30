# dicom2np

## usage

### local

```shell
$ python3 -m dicom2np \
    --data_dir ./data/dcm \
    --save_dir ./data/npz
```

### singularity

run

```shell
$ singularity run --bind ./data:/data rt4r.sif dicom2np --data_dir /data/dcm --save_dir /data/npz
```
