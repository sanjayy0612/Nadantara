import argparse
from pathlib import Path

from .audio import read_wav_chunks, write_wav
from .backends.identity import passthrough
from .config import AudioConfig
from .runtime import detect_runtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="seamtranslation")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect", help="show selected inference runtime")
    copy = commands.add_parser("passthrough", help="validate PCM audio transport")
    copy.add_argument("input", type=Path)
    copy.add_argument("output", type=Path)
    copy.add_argument("--chunk-ms", type=int, default=320)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        runtime = detect_runtime()
        print(f"device={runtime.device} torch_available={runtime.torch_available}")
        return 0
    config = AudioConfig(chunk_ms=args.chunk_ms)
    chunks = list(read_wav_chunks(args.input, config))
    write_wav(args.output, iter(passthrough(chunks)), config)
    duration = sum(len(c.pcm_s16le) for c in chunks) / (config.sample_rate * 2)
    print(f"wrote {len(chunks)} chunks ({duration:.3f}s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
