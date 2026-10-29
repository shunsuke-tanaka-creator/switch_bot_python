"""Three trainable LSTMs with bottom-up aggregation and explicit top-down fusion."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from .data import F, TEXT_DIM, ACTIVITIES


def encode(lstm, x, lengths):
    shape=x.shape[:-2]; seq=x.reshape(-1,x.shape[-2],x.shape[-1]); lens=lengths.reshape(-1)
    valid=lens>0
    out=x.new_zeros((len(lens),lstm.hidden_size))
    if valid.any():
        packed=pack_padded_sequence(seq[valid],lens[valid].cpu(),batch_first=True,enforce_sorted=False)
        _,(h,_)=lstm(packed);out[valid]=h[-1]
    return out.reshape(*shape,lstm.hidden_size)


class HierarchicalLSTM(nn.Module):
    def __init__(self, hidden=48, variant='full', use_text=True, adaptation=True):
        super().__init__()
        self.settings=dict(hidden=hidden,variant=variant,use_text=use_text,adaptation=adaptation)
        self.micro=nn.LSTM(F,hidden,batch_first=True)
        self.meso=nn.LSTM(hidden,hidden,batch_first=True)
        self.macro=nn.LSTM(hidden,hidden,batch_first=True)
        self.history_projection=nn.Linear(F,hidden)
        # Macro conditions Meso; both condition Micro. All are trained jointly.
        self.top_day=nn.Sequential(nn.Linear(hidden*2,hidden),nn.Tanh())
        self.top_work=nn.Sequential(nn.Linear(hidden*3,hidden),nn.Tanh())
        self.personal=nn.Sequential(nn.Linear(40,hidden),nn.Tanh())
        self.current=nn.Linear(hidden,5);self.daily=nn.Linear(hidden,5)
        self.activity=nn.Linear(hidden,len(ACTIVITIES))

    def forward(self,b):
        x=b['x']; history=b['history']
        if not self.settings['use_text']:
            x=x.clone();x[...,-TEXT_DIM:]=0;history=history.clone();history[...,-TEXT_DIM:]=0
        micro=encode(self.micro,x,b['lengths'])
        meso=encode(self.meso,micro,b['units'])
        macro_seq=torch.cat([self.history_projection(history),meso],dim=1)
        macro=encode(self.macro,macro_seq,b['days']+history.shape[1])
        batch=torch.arange(len(x),device=x.device);day=b['days']-1
        work=micro[batch,day,b['units'][batch,day]-1]
        daily=meso[batch,day]
        variant=self.settings['variant']
        if variant=='micro': daily=work;macro=torch.zeros_like(macro)
        elif variant=='micro_meso': macro=torch.zeros_like(macro)
        else: daily=self.top_day(torch.cat([daily,macro],-1))
        current=work if variant=='micro' else self.top_work(torch.cat([work,daily,macro],-1))
        if self.settings['adaptation']:
            person=self.personal(b['personal']);current=current+person;daily=daily+person
        return dict(current=1+4*torch.sigmoid(self.current(current)),daily=1+4*torch.sigmoid(self.daily(daily)),
                    activity=self.activity(current),embedding=current)
