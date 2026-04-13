# Stacked GRU decoder model as proposed in https://arxiv.org/abs/1802.04741

import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor
from .codes import LinearCode
from .utils import get_rank_zero_logger

log = get_rank_zero_logger(__name__)


class StackedGRU(nn.Module):

    def __init__(
        self,
        code: LinearCode,
        hidden_sz: int,
        n_layers: int = 1,
        n_steps: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        zero_padding: bool = False,
        output: str = "codeword",
    ) -> None:
        super().__init__()

        input_sz = code.n + code.m
        output_sz = code.k if output == "message" else code.n
        self.hidden_sz, self.output_sz, self.n_steps = hidden_sz, output_sz, n_steps
        self.zero_padding = zero_padding

        log.info(
            f"Building a {n_layers}-layer GRU decoder with hidden dimension {hidden_sz}"
        )
        log.info(
            f"The GRU decoder will process an input of length {self.n_steps} time steps"
        )
        if self.zero_padding:
            log.info("The decoder input is zeros at each time step but the first")
        else:
            log.info("The decoder input is repeated at each time step")

        # create layers
        self.gru = nn.GRU(
            input_sz, hidden_sz, n_layers, batch_first=True, bias=bias, dropout=dropout
        )
        self.to_logits = nn.Linear(hidden_sz, output_sz, bias=bias)

        # manually init weights, as pytorch's default was found to give weird results
        self.apply(self._init_weights)

        # for model summary
        self.example_input_array = torch.zeros(1, code.n), torch.zeros(1, code.m)

    # manually initialize weights for faster convergence (default init was found to give weird results)
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity="tanh")
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.kaiming_normal_(param.data, nonlinearity="tanh")
                elif "weight_hh" in name:
                    # nn.init.kaiming_normal_(param.data, nonlinearity='tanh')
                    sz = param.size(0)
                    nn.init.eye_(param.data[:, :sz])
                    nn.init.eye_(param.data[:, sz : 2 * sz])
                    nn.init.eye_(param.data[:, 2 * sz :])
                elif "bias" in name:
                    param.data.fill_(0.0)

    def forward(self, ym: Tensor, s: Tensor) -> Tensor:
        """Compact forward pass which processes all time steps in one call to torch.nn.GRU"""

        # input embedding
        x = torch.cat((ym, s), dim=1)
        if self.zero_padding:
            # pad input to shape (batch_size, n_steps, in_sz) with zeros
            x = torch.cat(
                (
                    x[:, None],
                    torch.zeros(x.size(0), 1, x.size(1), device=x.device).expand(
                        -1, self.n_steps - 1, -1
                    ),
                ),
                dim=1,
            )  # input = [x, 0, 0, ...] along dim=1
        else:
            # input is repeated at each time step, shape (batch_size, n_steps, in_sz)
            x = x.unsqueeze(1).expand(-1, self.n_steps, -1)

        # run gru on all time steps at once (faster than one step at a time)
        out = self.gru(x)[0]  # (B, L, H)

        # project the hidden state at the last time step to obtain the error pattern logits
        return self.to_logits(out[:, -1])


if __name__ == "__main__":
    pass
