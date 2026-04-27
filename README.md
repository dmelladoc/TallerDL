# Taller Deep learning - Ciencia de datos para la toma de decisiones

Autor: Dr. Diego Mellado Carreño
Fecha: 27 de Abril de 2025

---

## Introducción

En este repositiorio, vamos a disponibilizar los notebooks y otros elementos que requieren para ejecutar los ejemplos a revisar para la clase de taller de Deep Learning.

Este repositorio presenta los siguientes archivos:

- `load_databases.py`: Descarga las bases de datos a utilizar
- `Cats-v-dogs.ipynb`: Ejemplo de clasificación entre imágenes de Perros vs Gatos
- `CNN-medMNIST.ipynb`: Clasificación multiclase con imagenes médicas
- `UNet-Fishes.ipynb`: Segementación de imágenes

## Requisitos

Por defecto, este repositorio utiliza [`uv`](https://docs.astral.sh/uv/getting-started/installation/) para cargar las librerías y crear el ambiente de trabajo

- para instalar las librerías:

```sh
uv sync
```

Esto leerá el archivo `pyproject.toml` y cargará las librerias definidas.

Y para ejecutar el script de descarga de los datasets:

```sh
uv run load_databases.py
```

Ahora, si utilizan `anaconda` u otro gestor, deben instalar los siguientes paquetes:

- `numpy`
- `pandas`
- `matplotlib`
- `ipykernel`
- `scikit-learn`
- `pillow`
- `opencv-python`
- `pytorch`
- `torchvision`
- `torchmetrics`
- `torchinfo`
- `tqdm`
- `medmnist`
- `kagglehub`

Se deja disponible tambien un `requirements.txt` con las librerías, para su instalación con `pip`.

```sh
pip install -r requirements.txt
```

### Uso de CUDA en este proyecto

Dentro del `pyproject.toml` deben ajustar la ruta para que cargue las versiones con `CUDA`. Pues por defecto, está para `xpu`.
Deberan reemplazar la parte de:

```toml
[tool.uv.sources]
torch = [
    { index = "pytorch-xpu" },
]
torchvision = [
    { index = "pytorch-xpu" },
]
pytorch-triton-xpu = [
    { index = "pytorch-xpu" },
]

[[tool.uv.index]]
name = "pytorch-xpu"
url = "https://download.pytorch.org/whl/xpu"
explicit = true
```

con el siguiente extracto:

```toml
[tool.uv.sources]
torch = [
    { index = "pytorch-cu130" },
]
torchvision = [
    { index = "pytorch-cu130" },
]

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

y eliminar del `pyproject.toml` la librería `pytorch-triton-xpu<=3.5.0`.

## Descarga de datasets

para descargar los datasets, deberan ejecutar el script `load_databases.py`.

```sh
#utilizando uv
uv run load_databases.py --dataset [dataset] -o datasets/

#utilizando python directo
python load_databases.py --dataset [dataset] -o datasets/
```

donde:

- `--dataset` por defecto `all` (descarga todos) o `cats-vs-dogs`, `medmnist`, `fishes`.
- `-o`, `--output_dir` por defecto `datasets/`, o pueden definir la carpeta a instalar.

**NOTA**: Para descargar datasets desde Kaggle, requieren obtener una llave API desde su cuenta.
Esta llave deben ingresarla cuando se solicite el programa.
