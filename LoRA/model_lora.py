import math
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, RobertaConfig, RobertaModel
from peft import LoraConfig, get_peft_model


class PromptCS(nn.Module):
    def __init__(self, args, device, template):
        super(PromptCS, self).__init__()

        self.args = args
        self.mode = args.mode
        self.device = device
        self.use_lm_finetune = False

        # ============================================================
        # Load tokenizer
        # ============================================================
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path
        )

        # ============================================================
        # Set mode
        #
        # PromptCS:
        #   use soft prompt
        #
        # finetune:
        #   full parameter fine-tuning
        #
        # lora:
        #   LoRA fine-tuning, no soft prompt
        # ============================================================
        if args.mode == 'finetune':
            self.use_lm_finetune = True
            template = [0, 0]

        elif args.mode == 'lora':
            # LoRA does not use PromptCS soft prompt
            template = [0, 0]

        self.template = template

        # ============================================================
        # Set vocabulary
        # ============================================================
        self.vocab = self.tokenizer.get_vocab()

        # ------------------------------------------------------------
        # [PROMPT] is ONLY required by PromptCS.
        #
        # LoRA / finetune directly use normal code tokens.
        # ------------------------------------------------------------
        if self.mode == 'PromptCS':

            self.tokenizer.add_special_tokens(
                {
                    'additional_special_tokens': [
                        '[PROMPT]'
                    ]
                }
            )

            self.pseudo_token_id = \
                self.tokenizer.get_vocab()['[PROMPT]']

            self.prompt_tokens = [
                self.pseudo_token_id
            ]

        else:

            self.pseudo_token_id = None
            self.prompt_tokens = []

        # ============================================================
        # Special token IDs
        # ============================================================
        (
            self.pad_token_id,
            self.sep_token_id,
            self.eos_token_id,
            self.unk_token_id
        ) = self.get_special_token_id()

        self.sep_tokens = [
            self.sep_token_id
        ]

        self.eos_tokens = [
            self.eos_token_id
        ]

        # ============================================================
        # Load pre-trained language model
        # ============================================================
        self.model = create_model(
            self.args,
            self.use_lm_finetune
        )

        # ============================================================
        # Add LoRA
        # ============================================================
        if self.mode == 'lora':

            lora_config = LoraConfig(

                r=getattr(
                    args,
                    'lora_r',
                    8
                ),

                lora_alpha=getattr(
                    args,
                    'lora_alpha',
                    32
                ),

                lora_dropout=getattr(
                    args,
                    'lora_dropout',
                    0.05
                ),

                bias='none',

                task_type='CAUSAL_LM',

                target_modules=[
                    'c_attn',
                    'q_attn',
                    'c_proj'
                ]
            )

            self.model = get_peft_model(
                self.model,
                lora_config
            )

            # Print LoRA trainable parameter information
            self.model.print_trainable_parameters()

        # Move model to device
        self.model = self.model.to(
            self.device
        )

        # ============================================================
        # Freeze / train parameters
        # ============================================================

        # PromptCS:
        #     freeze language model
        #
        # finetune:
        #     train entire language model
        #
        # lora:
        #     PEFT automatically determines which LoRA
        #     parameters require gradients, so do NOT
        #     overwrite requires_grad here.
        # ============================================================
        if self.mode != 'lora':

            for param in self.model.parameters():

                param.requires_grad = \
                    self.use_lm_finetune

        # ============================================================
        # Get embedding layer
        # ============================================================
        self.embeddings = get_embedding_layer(
            self.args,
            self.model
        )

        # ============================================================
        # Prompt Encoder
        # ============================================================
        self.hidden_size = \
            self.embeddings.embedding_dim

        self.spell_length = \
            sum(self.template)

        # Prompt encoder is ONLY used by PromptCS
        if args.mode == 'PromptCS':

            self.prompt_agent = PromptAgent(

                self.template,

                self.hidden_size,

                self.tokenizer,

                self.device,

                args
            )

            self.prompt_agent = \
                self.prompt_agent.to(
                    self.device
                )

        # ============================================================
        # Sequence length settings
        # ============================================================
        self.max_target_length = \
            args.max_target_length

        self.max_code_length = \
            args.max_code_length

        self.lsm = nn.LogSoftmax(
            dim=-1
        )

        self.loss_fct = nn.CrossEntropyLoss(

            ignore_index=self.pad_token_id,

            reduction='sum'
        )


    # ================================================================
    # Special token ID
    # ================================================================
    def get_special_token_id(self):

        pad_token_id = None
        sep_token_id = None
        eos_token_id = None
        unk_token_id = None

        model_name = \
            self.args.model_name_or_path.lower()

        # ------------------------------------------------------------
        # StarCoder
        # ------------------------------------------------------------
        if 'starcoder' in model_name:

            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.unk_token_id
            )

            sep_token_id = \
                self.vocab['<fim_middle>']

            eos_token_id = \
                self.tokenizer.eos_token_id

            unk_token_id = \
                self.tokenizer.unk_token_id

        # ------------------------------------------------------------
        # PolyCoder
        # ------------------------------------------------------------
        elif 'polycoder' in model_name:

            pad_token_id = \
                self.vocab['<|padding|>']

            sep_token_id = \
                self.vocab['<|separator|>']

            eos_token_id = \
                self.vocab['<|endoftext|>']

            unk_token_id = \
                self.vocab['<|padding|>']

        # ------------------------------------------------------------
        # CodeGen
        # ------------------------------------------------------------
        elif 'codegen' in model_name:

            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.unk_token_id
            )

            sep_token_id = \
                self.vocab['//']

            eos_token_id = \
                self.tokenizer.eos_token_id

            unk_token_id = \
                self.tokenizer.unk_token_id

        return (
            pad_token_id,
            sep_token_id,
            eos_token_id,
            unk_token_id
        )


    # ================================================================
    # Embedding input
    # ================================================================
    def embed_input(self, queries):

        # PromptCS uses trainable soft prompt embedding
        if self.mode == 'PromptCS':

            return self.cstuning_embed_input(
                queries
            )

        # LoRA / full finetune directly use normal embeddings
        else:

            return self.finetune_embed_input(
                queries
            )


    # ================================================================
    # Normal embedding
    # Used by:
    #   finetune
    #   LoRA
    # ================================================================
    def finetune_embed_input(self, queries):

        return self.embeddings(
            queries
        )


    # ================================================================
    # PromptCS soft prompt embedding
    # ================================================================
    def cstuning_embed_input(self, queries):

        bz = queries.shape[0]

        queries_for_embedding = \
            queries.clone()

        # Replace [PROMPT] token IDs with unk token
        # before embedding lookup
        queries_for_embedding[
            (queries == self.pseudo_token_id)
        ] = self.unk_token_id

        raw_embeds = self.embeddings(
            queries_for_embedding
        )

        # Find positions of [PROMPT]
        blocked_indices = (
            queries == self.pseudo_token_id
        ).nonzero().reshape(
            (
                bz,
                self.spell_length,
                2
            )
        )[:, :, 1]

        # Generate soft prompt embeddings
        replace_embeds = \
            self.prompt_agent()

        # Replace embeddings
        for bidx in range(bz):

            for i in range(
                self.prompt_agent.spell_length
            ):

                raw_embeds[
                    bidx,
                    blocked_indices[bidx, i],
                    :
                ] = replace_embeds[i, :]

        return raw_embeds


    # ================================================================
    # Build query
    #
    # PromptCS:
    #
    #   [PROMPT] + CODE + [PROMPT] + SEP + TARGET
    #
    # LoRA / Finetune:
    #
    #   CODE + SEP + TARGET
    #
    # ================================================================
    def get_query(
        self,
        x_h,
        x_t=None
    ):

        # ------------------------------------------------------------
        # Code tokens
        # ------------------------------------------------------------
        code_tokens = \
            self.tokenizer.convert_tokens_to_ids(

                self.tokenizer.tokenize(
                    x_h
                )[:self.max_code_length]
            )

        # ------------------------------------------------------------
        # PromptCS
        # ------------------------------------------------------------
        if self.mode == 'PromptCS':

            left = (

                self.prompt_tokens
                * self.template[0]

                + code_tokens

                + self.prompt_tokens
                * self.template[1]
            )

        # ------------------------------------------------------------
        # LoRA / Finetune
        # ------------------------------------------------------------
        else:

            left = code_tokens

        # ------------------------------------------------------------
        # Target
        # ------------------------------------------------------------
        if x_t is not None:

            right = (

                self.tokenizer.convert_tokens_to_ids(

                    self.tokenizer.tokenize(
                        x_t
                    )[:self.max_target_length]
                )

                + self.eos_tokens
            )

        else:

            right = []

        # ------------------------------------------------------------
        # Final input
        # ------------------------------------------------------------
        input_ids = (

            left

            + self.sep_tokens

            + right
        )

        return (
            torch.LongTensor(
                input_ids
            ),
            len(left)
        )


    # ================================================================
    # Prepare model input
    # ================================================================
    def prepare_inputs(
        self,
        inputs
    ):

        # ------------------------------------------------------------
        # Padding
        # ------------------------------------------------------------
        inputs = pad_sequence(

            inputs,

            True,

            padding_value=self.pad_token_id

        ).long().to(
            self.device
        )

        # ------------------------------------------------------------
        # Attention mask
        # ------------------------------------------------------------
        attention_mask = \
            inputs != self.pad_token_id

        # ------------------------------------------------------------
        # Input embeddings
        # ------------------------------------------------------------
        inputs_embeds = \
            self.embed_input(
                inputs
            )

        inputs_embeds = \
            inputs_embeds.to(
                self.device
            )

        attention_mask = \
            attention_mask.to(
                self.device
            )

        # ------------------------------------------------------------
        # PromptCS / LoRA use FP16 language model
        #
        # Keep attention_mask boolean.
        # ------------------------------------------------------------
        if self.mode != 'finetune':

            inputs_embeds = \
                inputs_embeds.half()

        return (
            inputs,
            inputs_embeds,
            attention_mask
        )


    # ================================================================
    # Forward
    # ================================================================
    def forward(
        self,
        x_hs=None,
        x_ts=None
    ):

        bz = len(x_hs)

        # ============================================================
        # Training
        # ============================================================
        if x_ts is not None:

            inputs = []
            sum_idx = []
            ext_inputs = []

            for i in range(bz):

                input_ids, idx = \
                    self.get_query(
                        x_hs[i],
                        x_ts[i]
                    )

                inputs.append(
                    input_ids
                )

                sum_idx.append(
                    idx
                )

            # --------------------------------------------------------
            # Prepare model inputs
            # --------------------------------------------------------
            (
                inputs,
                inputs_embeds,
                attention_mask
            ) = self.prepare_inputs(
                inputs
            )

            # --------------------------------------------------------
            # Forward
            # --------------------------------------------------------
            output = self.model(

                inputs_embeds=inputs_embeds,

                attention_mask=attention_mask
            )

            logits = output.logits

            loss = None

            # --------------------------------------------------------
            # Calculate target loss
            # --------------------------------------------------------
            for i in range(bz):

                idx = sum_idx[i]

                shift_logits = \
                    logits[i][
                        idx:-1,
                        :
                    ].contiguous()

                shift_labels = \
                    inputs[i][
                        idx + 1:
                    ].contiguous()

                if loss is None:

                    loss = self.loss_fct(

                        shift_logits.view(
                            -1,
                            shift_logits.size(-1)
                        ),

                        shift_labels.view(
                            -1
                        )
                    )

                else:

                    loss += self.loss_fct(

                        shift_logits.view(
                            -1,
                            shift_logits.size(-1)
                        ),

                        shift_labels.view(
                            -1
                        )
                    )

            loss = loss / bz

            return loss

        # ============================================================
        # Generation / Evaluation
        # ============================================================
        else:

            queries = []
            sum_idx = []
            tmp_idx = []

            for i in range(bz):

                query, idx = \
                    self.get_query(
                        x_h=x_hs[i]
                    )

                queries.append(
                    query
                )

                sum_idx.append(
                    idx
                )

                tmp_idx.append(
                    idx
                )

            # --------------------------------------------------------
            # Auto-regressive generation
            # --------------------------------------------------------
            for _ in range(
                self.max_target_length
            ):

                (
                    inputs,
                    inputs_embeds,
                    attention_mask
                ) = self.prepare_inputs(
                    queries
                )

                output = self.model(

                    inputs_embeds=inputs_embeds,

                    attention_mask=attention_mask
                )

                logits = output.logits

                for i in range(bz):

                    idx = tmp_idx[i]

                    tmp_idx[i] += 1

                    next_token_logits = \
                        logits[
                            i,
                            idx:idx + 1,
                            :
                        ]

                    _, next_token = \
                        torch.max(
                            next_token_logits,
                            dim=1
                        )

                    queries[i] = torch.cat(
                        [
                            queries[i].to(
                                self.device
                            ),
                            next_token
                        ],
                        dim=0
                    )

            # --------------------------------------------------------
            # Decode result
            # --------------------------------------------------------
            answer = []

            for i in range(bz):

                idx = sum_idx[i]

                t = queries[i][
                    idx + 1:
                ]

                t = t.tolist()

                if self.eos_token_id in t:

                    t = t[
                        :t.index(
                            self.eos_token_id
                        )
                    ]

                words = \
                    self.tokenizer.decode(
                        t
                    ).replace(
                        '\n',
                        ''
                    )

                answer.append(
                    words
                )

            return answer



# ====================================================================
# Prompt Agent
# ====================================================================
class PromptAgent(torch.nn.Module):

    def __init__(
        self,
        template,
        hidden_size,
        tokenizer,
        device,
        args
    ):

        super().__init__()

        self.device = device

        self.spell_length = \
            sum(template)

        self.hidden_size = \
            hidden_size

        self.embed_size = \
            hidden_size

        self.tokenizer = \
            tokenizer

        self.args = args

        # ============================================================
        # Prompt embedding
        # ============================================================
        self.cloze_length = \
            template

        self.cloze_mask = [

            [1] * self.cloze_length[0]

            +

            [1] * self.cloze_length[1]
        ]

        self.cloze_mask = \
            torch.LongTensor(
                self.cloze_mask
            ).bool().to(
                self.device
            )

        self.seq_indices = \
            torch.LongTensor(

                list(
                    range(
                        len(
                            self.cloze_mask[0]
                        )
                    )
                )

            ).to(
                self.device
            )

        # ============================================================
        # Prompt Encoder
        # ============================================================

        if args.prompt_encoder_type == "lstm":

            self.prompt_encoder = \
                Encoder_BiLSTM(

                    input_size=self.hidden_size,

                    hidden_size=
                    self.hidden_size // 2,

                    num_layers=2,

                    dropout=0.0,

                    bidirectional=True,

                    batch_first=True
                )

        elif args.prompt_encoder_type == "transformer":

            self.prompt_encoder = \
                Encoder_Transformer(

                    d_model=self.hidden_size,

                    nhead=8,

                    num_layers=6,

                    max_len=len(
                        self.cloze_mask[0]
                    )
                )

        # ============================================================
        # Prompt Embedding Layer
        # ============================================================
        self.embedding = \
            torch.nn.Embedding(

                len(
                    self.cloze_mask[0]
                ),

                self.embed_size

            ).to(
                self.device
            )

        # ============================================================
        # MLP Head
        # ============================================================
        self.mlp_head = nn.Sequential(

            nn.Linear(
                self.hidden_size,
                self.hidden_size
            ),

            nn.ReLU(),

            nn.Linear(
                self.hidden_size,
                self.hidden_size
            )
        )

        print(
            "init prompt encoder..."
        )


    # ================================================================
    # Forward
    # ================================================================
    def forward(self):

        input_embeds = \
            self.embedding(

                self.seq_indices

            ).unsqueeze(
                0
            )

        output_embeds = \
            self.prompt_encoder(
                input_embeds
            )

        output_embeds = \
            self.mlp_head(
                output_embeds
            ).squeeze()

        return output_embeds



# ====================================================================
# BiLSTM Prompt Encoder
# ====================================================================
class Encoder_BiLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout,
        bidirectional,
        batch_first
    ):

        super(
            Encoder_BiLSTM,
            self
        ).__init__()

        self.lstm_head = \
            torch.nn.LSTM(

                input_size=input_size,

                hidden_size=hidden_size,

                num_layers=num_layers,

                dropout=dropout,

                bidirectional=bidirectional,

                batch_first=batch_first
            )


    def forward(
        self,
        inputs
    ):

        outputs = \
            self.lstm_head(
                inputs
            )[0]

        return outputs



# ====================================================================
# Transformer Prompt Encoder
# ====================================================================
class Encoder_Transformer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        max_len
    ):

        super(
            Encoder_Transformer,
            self
        ).__init__()

        self.encoder_layer = \
            nn.TransformerEncoderLayer(

                d_model=d_model,

                nhead=nhead
            )

        self.encoder = \
            nn.TransformerEncoder(

                self.encoder_layer,

                num_layers=num_layers
            )

        self.pos_embedding = \
            PositionalEncoding(

                d_model,

                0.1,

                max_len
            )


    def forward(
        self,
        inputs
    ):

        input_embedding = \
            self.pos_embedding(
                inputs
            )

        input_embedding = \
            input_embedding.permute(
                1,
                0,
                2
            )

        outputs = \
            self.encoder(
                input_embedding
            ).permute(
                [
                    1,
                    0,
                    2
                ]
            )

        return outputs



# ====================================================================
# Positional Encoding
# ====================================================================
class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        dropout=0.1,
        max_len=2501
    ):

        super(
            PositionalEncoding,
            self
        ).__init__()

        self.dropout = \
            nn.Dropout(
                p=dropout
            )

        pe = torch.zeros(
            max_len,
            d_model
        )

        position = torch.arange(

            0,

            max_len,

            dtype=torch.float

        ).unsqueeze(
            1
        )

        div_term = torch.exp(

            torch.arange(

                0,

                d_model,

                2

            ).float()

            *

            (
                -math.log(
                    10000.0
                )
                /
                d_model
            )
        )

        pe[:, 0::2] = \
            torch.sin(
                position * div_term
            )

        pe[:, 1::2] = \
            torch.cos(
                position * div_term
            )

        pe = pe.unsqueeze(
            0
        )

        self.register_buffer(
            'pe',
            pe
        )


    def forward(
        self,
        x
    ):

        x = x + self.pe

        return self.dropout(
            x
        )



# ====================================================================
# Create Language Model
# ====================================================================
def create_model(
    args,
    use_lm_finetune
):

    model = \
        AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path
        )

    # PromptCS and LoRA use half precision
    if not use_lm_finetune:

        model = \
            model.half()

    return model



# ====================================================================
# Get Embedding Layer
# ====================================================================
def get_embedding_layer(
    args,
    model
):

    return \
        model.base_model.get_input_embeddings()