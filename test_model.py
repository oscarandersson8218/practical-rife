import argparse
from pathlib import Path

import cv2
import torch

from train_log.RIFE_HDv3 import Model


DEFAULT_PAIRS = (
    ("I0", "demo/I0_0.png", "demo/I0_1.png"),
    ("i", "demo/i0.png", "demo/i1.png"),
    ("I2", "demo/I2_0.png", "demo/I2_1.png"),
)


def read_image(path, width, height, device):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{path} must be a 3-channel image")

    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().div(255.0)
    return tensor.unsqueeze(0).to(device)


def write_image(path, tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor[0].detach().clamp(0.0, 1.0).mul(255.0).byte()
    image = image.cpu().numpy().transpose(1, 2, 0)
    cv2.imwrite(str(path), image)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export RIFE and run midpoint inference on demo image pairs."
    )
    parser.add_argument("--model", type=Path, default=Path("train_log"), help="directory with flownet.pkl")
    parser.add_argument("--export", type=Path, default=Path("output/test_model/rife_inference.pt2"))
    parser.add_argument("--output", type=Path, default=Path("output/test_model"))
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=768)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cpu"
    model = Model()
    model.load_model(args.model)
    model.eval()
    model.to(device)

    example_inputs = model.example_inputs(args.width, args.height)
    with torch.no_grad():
        exported_program = torch.export.export(model, example_inputs)
    args.export.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported_program, args.export)
    graph_module = exported_program.module()

    outputs = []
    for name, img0_path, img1_path in DEFAULT_PAIRS:
        img0 = read_image(img0_path, args.width, args.height, device)
        img1 = read_image(img1_path, args.width, args.height, device)
        with torch.no_grad():
            middle = graph_module(img0, img1)
        output_path = args.output / f"{name}_middle.png"
        write_image(output_path, middle)
        outputs.append(output_path)

    print(f"Exported: {args.export}")
    for output in outputs:
        print(f"Wrote: {output}")

if __name__ == "__main__":
    main()
