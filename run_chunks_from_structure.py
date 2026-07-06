import argparse
from pageindex.page_index import build_chunks_from_existing_structure
from pageindex.utils import ConfigLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", required=True)
    parser.add_argument("--structure_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens-per-node", type=int, default=None)
    parser.add_argument("--updated_structure_output", default=None)

    args = parser.parse_args()

    opt = ConfigLoader().load({
        k: v for k, v in {
            "model": args.model,
            "max_token_num_each_node": args.max_tokens_per_node,
        }.items()
        if v is not None
    })

    build_chunks_from_existing_structure(
        pdf_path=args.pdf_path,
        structure_path=args.structure_path,
        output_path=args.output,
        opt=opt,
        updated_structure_path=args.updated_structure_output
    )

    print(f"Chunks saved to: {args.output}")