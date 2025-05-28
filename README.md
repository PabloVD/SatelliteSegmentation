Install [GDAL](https://gdal.org/en/stable/)

```sh
 sudo apt install libgdal-dev
 ```

```sh
pip install --no-build-isolation --no-cache-dir --force-reinstall gdal 
```

```sh
conda install -c conda-forge gdal
```

Add venv kernel:
```sh
ipython kernel install --user --name=segmenter
```

```sh
conda install -c conda-forge localtileserver
```