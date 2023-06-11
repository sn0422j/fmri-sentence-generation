import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.joinpath("step3_evaluation")))
from my_metrics import compute_bert_score, compute_blue

def main():
    for text_true, text_pred in [
        [
            "a dress is a piece of clothing that women sometimes wear.",
            "the tree is a regular sized person and makes boots, but has a black nose.",
        ],
        [
            "a horse has a tail and a mane on its neck, and is usually gray or brown.",
            "an animal doesn't like a roundhouse or bamboo, in which there are twigs and trees.",
        ],
        [
            "the blades are usually metallic, with plastic or metal handles.",
            "some knife blades are fashioned with pieces of dough, and then put in a silver barrel.",
        ],
        [
            "spiders are insects with eight legs that make silk.",
            "a worm is a complete body suit with a <unk>.",
        ],
    ]:
        print(f"{text_true}\n   -> {text_pred}\n")
        print(f"BLUE: {compute_blue([text_true], [text_pred]):.3f}")
        print(f"BERTScore: {compute_bert_score([text_true], [text_pred]):.3f}")


if __name__ == "__main__":
    main()
