import argparse
import json
import pathlib


def truncate_json(input_path, output_path, num_samples):
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    truncated = data[:num_samples]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(truncated, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(truncated)} samples to {output_path}")


def chunk_json(input_path, output_dir, chunk_size):
    input_path = pathlib.Path(input_path)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    for i in range(0, total, chunk_size):
        chunk = data[i:i + chunk_size]
        out_file = output_dir / f"chunk_{i//chunk_size:03d}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(chunk)} samples to {out_file}")
    print(f"Total: {total} samples; chunk size: {chunk_size}")


def main():
    parser = argparse.ArgumentParser(description="Truncate or chunk JSON dataset")
    parser.add_argument("input", help="输入 JSON 文件")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_trunc = subparsers.add_parser("truncate", help="截取前 N 条")
    p_trunc.add_argument("output", help="输出 JSON 文件")
    p_trunc.add_argument("num", type=int, help="截取条数")

    p_chunk = subparsers.add_parser("chunk", help="按块切分")
    p_chunk.add_argument("output_dir", help="输出目录，生成 chunk_000.json 等")
    p_chunk.add_argument("chunk_size", type=int, help="每块条数，例如 200000")

    args = parser.parse_args()

    if args.mode == "truncate":
        truncate_json(args.input, args.output, args.num)
    elif args.mode == "chunk":
        chunk_json(args.input, args.output_dir, args.chunk_size)


if __name__ == "__main__":
    main()
