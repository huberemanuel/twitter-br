from transformers import RobertaConfig


def main():
    # TODO: Parametrize things
    config = RobertaConfig.from_pretrained("roberta-large", vocab_size=50265)
    config.save_pretrained("./ttbr-roberta-large")


if __name__ == "__main__":
    main()
