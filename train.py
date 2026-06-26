import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from dotenv import load_dotenv
import os
from tqdm import tqdm
import warnings

import sys
from pathlib import Path

from model import build_transformer
from data import create_data_loader, train_tokenizers
from config import ConfigLoader
from helpers.common import latest_weights_file_path

load_dotenv(dotenv_path=".env", override=True)
hf_token = os.getenv("HF_TOKEN")


def prepare_model_data(
    config: dict,
    hf_token: str | None,
):
    trained_tokens_dict = train_tokenizers(
        dataset_path=config["dataset_path"],
        dataset_name=config["dataset_name"],
        source_language=config["source_language"],
        target_language=config["target_language"],
        tokenizer_dir=config["tokenizer_dir"],
        max_seq_len=config["max_seq_len"],
        hf_token=hf_token,
    )

    train_data_loader = create_data_loader(
        dataset_path=config["dataset_path"],
        dataset_name=config["dataset_name"],
        split="train",
        trained_tokens_dict=trained_tokens_dict,
        source_language=config["source_language"],
        target_language=config["target_language"],
        batch_size=config["train_batch_size"],
        shuffle=True,
        hf_token=hf_token,
    )

    val_data_loader = create_data_loader(
        dataset_path=config["dataset_path"],
        dataset_name=config["dataset_name"],
        split="validation",
        trained_tokens_dict=trained_tokens_dict,
        source_language=config["source_language"],
        target_language=config["target_language"],
        batch_size=config["val_batch_size"],
        shuffle=True,
        hf_token=hf_token,
    )

    model = build_transformer(
        source_vocab_size=trained_tokens_dict["source_vocab_size"],
        target_vocab_size=trained_tokens_dict["target_vocab_size"],
        source_seq_len=trained_tokens_dict["source_seq_len"],
        target_seq_len=trained_tokens_dict["target_seq_len"],
        d_model=config["d_model"],
        h=config["num_heads"],
        dropout=config["dropout"],
        n_encoder=config["n_encoder"],
        n_decoder=config["n_decoder"],
    )

    return model, train_data_loader, val_data_loader, trained_tokens_dict


def greedy_decoder(model, target_tokenizer, encoder_input, encoder_mask, device, max_translation_len):
    sos = target_tokenizer.token_to_id("<SOS>")
    eos = target_tokenizer.token_to_id("<EOS>")

    encoder_output = model.encode(encoder_input, encoder_mask)

    # translation_token_ids_so_far
    decoder_input = torch.empty(1, 1).fill_(sos).type_as(encoder_input).to(device) # (B=1, target_seq_len=1)

    last_word_token_id = None
    while last_word_token_id != eos and decoder_input.shape[1] < max_translation_len:
        # There is no padding mask; **padding to the encoder intput is only necessary for B > 1 since all batch entries should have same dim**
        causal_mask = torch.tril(torch.ones(decoder_input.shape[1], decoder_input.shape[1])).type_as(encoder_mask).to(device) 
        decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, causal_mask)
        projection = model.project(decoder_output) # (B, target_seq_len, target_vocab_size) **target_seq_len growing here**

        last_word_logits = projection[:, -1] # equivalent of projection[:, -1, :] # (B=1, target_vocab_size)

        # This assumes token id is the same as the index in vocabulary
        # This works correctly because nn.Embedding and the projection layer are both indexed by token ID
        _, last_word_token_id = torch.max(last_word_logits, dim=-1) # get the index of the vocab with largest logit

        decoder_input = torch.cat([decoder_input, last_word_token_id.view(1, 1).type_as(decoder_input).to(device)], dim=1)

    return decoder_input.squeeze(0)



def run_validation(model, data_loader, target_tokenizer, target_seq_len, device, print_msg, writer, global_step, max_num_val_steps=None):
    model.eval()

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80
    
    translation_sentence_ls, target_sentence_ls = [], []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            assert encoder_input.shape[0] == 1, "Validitaion batch size must be 1"
            
            translation_token_ids = greedy_decoder(  # shape: (translation_seq_len)
                model=model,
                target_tokenizer=target_tokenizer,
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
                device=device,
                max_translation_len=target_seq_len
            )
            translation_sentence = target_tokenizer.decode(translation_token_ids.detach().cpu().numpy())
            target_sentence = batch["target_sentence"][0]
            source_sentence = batch["source_sentence"][0]
            
            translation_sentence_ls.append(translation_sentence)
            target_sentence_ls.append(target_sentence_ls)

            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_sentence}")
            print_msg(f"{f'TARGET: ':>12}{target_sentence}")
            print_msg(f"{f'TRANSLATION: ':>12}{translation_sentence}")

            if max_num_val_steps is not None and i >= max_num_val_steps:
                print_msg('='*console_width)
                break

    if writer:
        # Log the character error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(translation_sentence, target_sentence)
        writer.add_scalar("validation CER", cer, global_step)
        writer.flush()

        # Log the word error rate 
        metric = torchmetrics.WordErrorRate()
        wer = metric(translation_sentence, target_sentence)
        writer.add_scalar("validation WER", wer, global_step)
        writer.flush()

        # Log the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(translation_sentence, target_sentence)
        writer.add_scalar("validation BLEU", bleu, global_step)
        writer.flush()



def train_model(
    config_file_path: str,
    hf_token: str = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if (device == "cuda"):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    torch.device(device)
    
    config = ConfigLoader.from_yaml(config_file_path)
    model, train_data_loader, val_data_loader, trained_tokens_dict = prepare_model_data(config=config, hf_token=hf_token)
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=trained_tokens_dict["source_tokenizer"].token_to_id('<PAD>'),
        label_smoothing=0.1,
    )

    # Make sure the paths are created
    Path(f"{config['model_weights_dir']}").mkdir(parents=True, exist_ok=True)
    Path(f"{config['tensorboard_log_dir']}").mkdir(parents=True, exist_ok=True)

    global_step = 0
    # Tensorboard
    writer = SummaryWriter(config["tensorboard_log_dir"])
    
    model_filename = latest_weights_file_path(config) if config["preload"] == "latest" else None
    initial_epoch = 0
    if model_filename:
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        print("No model to preload, starting from scratch")

    for epoch in range(initial_epoch, config["num_epochs"]):
        batch_iterator = tqdm(train_data_loader, desc=f"Processing Epoch {epoch:02d}")
        model.train()
        for i, batch in enumerate(batch_iterator):
            
            encoder_output = model.encode(
                batch["encoder_input"].to(device),
                batch["encoder_mask"].to(device),
            )
            decoder_output = model.decode(
                batch["decoder_input"].to(device),
                encoder_output,
                batch["encoder_mask"].to(device),
                batch["decoder_mask"].to(device),
            )
            projection = model.project(decoder_output)

            label = batch["label"].to(device).view(-1)
            loss = loss_fn(projection.view(-1, trained_tokens_dict["target_vocab_size"]), label)

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(
            model=model,
            data_loader=val_data_loader,
            target_tokenizer=trained_tokens_dict["target_tokenizer"],
            target_seq_len=trained_tokens_dict["target_seq_len"],
            device=device,
            print_msg=lambda msg: batch_iterator.write(msg),
            writer=writer,
            global_step=global_step,
            max_num_val_steps=config["max_num_val_steps"],
        )

        model_file_name = f"{config['model_weights_dir']}/{config['model_basename']}_epoch_{epoch}.pt"
        torch.save(
            obj={
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "global_step": global_step,
            },
            f=model_file_name,
        )


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config_file_path = "./config/parameters.yaml"
    train_model(config_file_path=config_file_path)