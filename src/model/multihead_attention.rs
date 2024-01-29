use candle_nn::{Module, Dropout};
use candle_core::{Tensor, Shape, Device};
use candle_transformers::models::with_tracing::{linear, linear_no_bias, Linear};
use candle_nn::{self, embedding, Conv1d, Conv1dConfig, Embedding, LayerNorm, VarBuilder};

use std::{collections::HashMap, ops::{Mul, Not}};
use anyhow::Result;
use uuid;


#[derive(Debug, Clone)]
struct RotaryEmbedding;

impl RotaryEmbedding{
    fn load(dim: usize) -> Self{
        Self
    }
}

// Define the MultiheadAttention struct
pub struct MultiheadAttention {
    embed_dim: usize,
    kdim: usize,
    vdim: usize,
    qkv_same_dim: bool,
    num_heads: usize,
    dropout: f32,
    head_dim: usize,
    scaling: f32,
    self_attention: bool,
    encoder_decoder_attention: bool,
    k_proj: Linear,
    v_proj: Linear,
    q_proj: Linear,
    out_proj: Linear,
    bias_k: Option<Tensor>,
    bias_v: Option<Tensor>,
    add_zero_attn: bool,
    rot_emb: Option<RotaryEmbedding>,
    device: Device,
    // enable_torch_version: bool,
}

impl MultiheadAttention {
    pub fn load(
        embed_dim: usize, 
        num_heads: usize, 
        kdim: Option<usize>, vdim: Option<usize>, 
        dropout: f32, 
        bias: bool, 
        add_bias_kv: bool,
        add_zero_attn: bool,
        self_attention: bool,
        encoder_decoder_attention: bool,
        use_rotary_embeddings: bool,
        vb: VarBuilder,
        device: Device
    ) -> Result<Self>  {
        let head_dim = embed_dim / num_heads;
        assert!(embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads");

        let span = tracing::span!(tracing::Level::TRACE, "multi-head-attn");
        let softmax_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-softmax");
        let matmul_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-matmul");
    
        // let embed_dim = n_state;
        let span = tracing::span!(tracing::Level::TRACE, "multi-head-attn");
        let softmax_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-softmax");
        let matmul_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-matmul");
        let q_proj = linear(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let v_proj = linear(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let k_proj = linear(embed_dim, embed_dim, vb.pp("k_proj"))?;
        let out_proj = linear(embed_dim, embed_dim, vb.pp("out_proj"))?;
        let head_dim = embed_dim / num_heads;
        let scaling = (head_dim as f32).powf(-0.5);

        let (bias_k, bias_v) =  if add_bias_kv{
             (Some(vb.get((1, 1, embed_dim), "bias_k")?),
             Some(vb.get((1, 1, embed_dim), "bias_k")?))
        }else{
            (None, None)
        };
        
        let kdim2 = if let Some(kdim) = kdim {
            kdim
        }else{
            embed_dim
        };
        let vdim2 = if let Some(vdim) = vdim {
            vdim
        }else{
            embed_dim
        };

        let qkv_same_dim = {(kdim2 == embed_dim) & (vdim2 == embed_dim)};

        // let rot_emb = None;
        let rot_emb = if use_rotary_embeddings{
            Some(RotaryEmbedding::load(head_dim))
        }else{
            None
        };
            


        Ok(Self {
            embed_dim: todo!(),
            kdim: kdim2,
            vdim: vdim2,
            qkv_same_dim,
            num_heads,
            dropout,
            head_dim,
            scaling,
            self_attention,
            encoder_decoder_attention,
            k_proj,
            v_proj,
            q_proj,
            out_proj,
            bias_k,
            bias_v,
            add_zero_attn,
            rot_emb,
            device
        })
    }

    fn default(embed_dim:usize, num_heads:usize, vb: VarBuilder) -> Result<Self>{
        // kdim=None,
        // vdim=None,
        // dropout=0.0,
        // bias=True,
        // add_bias_kv: bool = False,
        // add_zero_attn: bool = False,
        // self_attention: bool = False,
        // encoder_decoder_attention: bool = False,
        // use_rotary_embeddings: bool = False,

        Self::load(embed_dim, num_heads, 
            None, None, 0.0, true, false, 
            false, false, false, false, vb, Device::Cpu)
    }

    // Implement the forward function
    pub fn forward(&self, 
        query: &Tensor, 
        key: Option<Tensor>, 
        value: Option<Tensor>, 
        key_padding_mask: &Option<Tensor>,
        incremental_state:Option<HashMap<String, HashMap<String, Option<Tensor>>>>,
        mut need_weights: bool,
        static_kv: bool,
        attn_mask: Option<Tensor>,
        before_softmax: bool,
        need_head_weights: bool
    ) -> Result<()> {
        if need_head_weights{
            need_weights = true;
        }
        let mut key = key;
        let mut value = value;
        // let (tgt_len, bsz, embed_dim) = 
        let shape = query.shape();
        let (tgt_len, bsz, embed_dim) = shape.dims3()?;

        // if incremental_state is not None:
        //     saved_state = self._get_input_buffer(incremental_state)
        //     if saved_state is not None and "prev_key" in saved_state:
        //         # previous time steps are cached - no need to recompute
        //         # key and value if they are static
        //         if static_kv:
        //             assert self.encoder_decoder_attention and not self.self_attention
        //             key = value = None
        // else:
        //     saved_state = None
        let saved_state = self.get_input_buffer(&incremental_state);
        if saved_state.contains_key("prev_key") && static_kv {
            key = None;
            value = None;
        }

        let (mut q, k, v ) = if self.self_attention{
            (
                Some(self.q_proj.forward(query)?), 
                Some(self.k_proj.forward(query)?), 
                Some(self.v_proj.forward(query)?)
            )
        }else if self.encoder_decoder_attention{
            let q = Some(self.q_proj.forward(query)?);
            let (mut k, mut v) = (None, None);
            if let Some(key) = key{
                if let Some(value) = value{
                    k = Some(self.k_proj.forward(query)?);
                    v = Some(self.v_proj.forward(query)?);
                }
            }
            (q, k, v)
        }else{
            let q = Some(self.q_proj.forward(query)?);
            let (mut k, mut v) = (None, None);
            if let Some(key) = key{
                if let Some(value) = value{
                    k = Some(self.k_proj.forward(&key)?);
                    v = Some(self.v_proj.forward(&value)?);
                }
            }
            (q, k, v)
        };

        // q *= self.scaling
        let q = q.map(|x| x.matmul(
            &Tensor::new(self.scaling, &self.device).unwrap()
        ).unwrap());

        // if self.bias_k is not None:
        //     assert self.bias_v is not None
        //     k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
        //     v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
        //     if attn_mask is not None:
        //         attn_mask = torch.cat(
        //             [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
        //         )
        //     if key_padding_mask is not None:
        //         key_padding_mask = torch.cat(
        //             [
        //                 key_padding_mask,
        //                 key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
        //             ],
        //             dim=1,
        //         )

        // q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        // if k is not None:
        //     k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        // if v is not None:
        //     v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        let q = q.unwrap().reshape((tgt_len, bsz * self.num_heads, self.head_dim))?.transpose(0, 1)?;
        let k = if let Some(k) = k{
            let k_len = k.dims().iter().product::<usize>();
            let dim1 = k_len/bsz * self.num_heads / self.head_dim;
            let k = k.reshape((dim1, bsz * self.num_heads, self.head_dim))?.transpose(0, 1)?;
            Some(k)
        }else{None};

        let v = if let Some(v) = v{
            let v_len = v.dims().iter().product::<usize>();
            let dim1 = v_len/bsz * self.num_heads / self.head_dim;
            let v = v.reshape((dim1, bsz * self.num_heads, self.head_dim))?.transpose(0, 1)?;
            Some(v)
        }else{None};

        // if saved_state.
        let k = if let Some(_prev_key) = saved_state.get("prev_key"){
            let _prev_key = _prev_key.unwrap();
            let k_len = _prev_key.dims().iter().product::<usize>();
            let dim2 = k_len / bsz * self.num_heads / self.head_dim;
            let prev_key = _prev_key.reshape((bsz * self.num_heads, dim2, self.head_dim))?;
            if static_kv {
                Some(prev_key)
            } else { 
                Some(Tensor::cat(&[prev_key, k.unwrap()], 1)? )
            }
        }else{None};

        let v = if let Some(_prev_value) = saved_state.get("prev_value"){
            let _prev_value = _prev_value.unwrap();
            let v_len = _prev_value.dims().iter().product::<usize>();
            let dim2 = v_len / bsz * self.num_heads / self.head_dim;
            let prev_value = _prev_value.reshape((bsz * self.num_heads, dim2, self.head_dim))?;
            if static_kv {Some(prev_value)} else { Some(Tensor::cat(&[prev_value, v.unwrap()], 1)?) }
        }else{None};
        
        let prev_key_padding_mask = saved_state.get("prev_key_padding_mask");
        
        //     prev_key_padding_mask: Optional[Tensor] = None
        //     if "prev_key_padding_mask" in saved_state:
        //         prev_key_padding_mask = saved_state["prev_key_padding_mask"]
        //     assert k is not None and v is not None
        //     key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
        //         key_padding_mask=key_padding_mask,
        //         prev_key_padding_mask=prev_key_padding_mask,
        //         batch_size=bsz,
        //         src_len=k.size(1),
        //         static_kv=static_kv,
        //     )

        //     saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
        //     saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
        //     saved_state["prev_key_padding_mask"] = key_padding_mask
        //     # In this branch incremental_state is never None
        //     assert incremental_state is not None
        //     incremental_state = self._set_input_buffer(incremental_state, saved_state)
        // assert k is not None
        // src_len = k.size(1)




            Ok(())

        }
        

    fn get_full_incremental_state_key(self, key: &str) -> &str{
        
        return &format!("{}.{}", uuid::Uuid::new_v4(), key)
    }

    ///Helper for getting incremental state for an nn.Module.
    fn get_incremental_state(self, 
        incremental_state: &Option<HashMap<String, HashMap<String, Option<Tensor>>>>, 
        key: &str
    ) -> Option<HashMap<String, Option<Tensor>>>{
        let full_key = self.get_full_incremental_state_key(key);
        if let Some(incremental_state) = incremental_state{
            if incremental_state.contains_key(full_key).not(){
                None
            }else{
                incremental_state.get(full_key).cloned()
            }
        }else{
            None
        }
    }


    fn get_input_buffer(self, 
        incremental_state: &Option<HashMap<String, HashMap<String, Option<Tensor>>>>
    ) -> HashMap<String, Option<Tensor>>{
        let result = self.get_incremental_state(incremental_state, "attn_state");
        if let Some(result) = result{
            result
        }else{
            HashMap::default()
        }


    }
    // Implement additional methods as required
 }
// Additional utility functions as needed
