import os
import zipfile
from argparse import ArgumentParser

import kagglehub
from medmnist import ChestMNIST


def load_cats_and_dogs(path):
    # descargamos los datos usando kagglehub
    kagglehub.login()

    datapath = kagglehub.competition_download("dogs-vs-cats")

    # creamos una carpeta para guardar los datos
    outpath = os.path.join(path, "cats-vs-dogs")
    os.makedirs(outpath, exist_ok=True)

    # extraemos el contenido a la carpeta de salida
    ## Entrenamiento
    with zipfile.ZipFile(os.path.join(datapath, "train.zip"), "r") as zip_ref:
        zip_ref.extractall(outpath)
    ## test
    with zipfile.ZipFile(os.path.join(datapath, "test1.zip"), "r") as zip_ref:
        zip_ref.extractall(outpath)


def load_medmnist(path):
    # se descarga automáticamente
    outpath = os.path.join(path, "medmnist")
    os.makedirs(outpath, exist_ok=True)

    # para cada split, se descarga el dataset correspondiente
    for split in ["train", "val", "test"]:
        dataset = ChestMNIST(
            split=split,
            download=True,
            size=224,
            as_rgb=True,
            root=outpath,
        )


def load_fishes(path):
    kagglehub.login()

    outpath = os.path.join(path, "fishes")
    os.makedirs(outpath, exist_ok=True)
    datapath = kagglehub.dataset_download(
        "crowww/a-large-scale-fish-dataset", output_dir=outpath
    )


def main(args):
    if args.dataset in ["cats-vs-dogs", "all"]:
        print("Descargando el dataset Cats vs Dogs...")
        load_cats_and_dogs(args.output_dir)

    if args.dataset in ["medmnist", "all"]:
        print("Descargando el dataset MedMNIST...")
        load_medmnist(args.output_dir)

    if args.dataset in ["fishes", "all"]:
        print("Descargando el dataset de fishes...")
        load_fishes(args.output_dir)

    lista_instalados = ", ".join(args.dataset)
    print(
        f"Descarga de datasets completada: {lista_instalados}\nEstos se encuentran en el directorio: {args.output_dir}"
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cats-vs-dogs", "medmnist", "fishes", "all"],
        default="all",
        help="Dataset a descargar. Por defecto, se descargarán todos los datasets.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="datasets/",
        help="Directorio donde se guardarán los datasets descargados (los de kagglehub).",
    )
    args = parser.parse_args()
    main(args)
