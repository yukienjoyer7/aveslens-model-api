from training.dataset import export_label_map, load_split


def main(output_path: str = "labels.json") -> None:
    dataset = load_split("train")
    export_label_map(dataset, output_path)
    print(f"labels saved → {output_path}")


if __name__ == "__main__":
    main()
