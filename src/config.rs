use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize, Debug)]
struct Config {
    #[serde(rename = "_name_or_path")]
    name_or_path: String,
    architectures: Vec<String>,
    attention_probs_dropout_prob: f64,
    classifier_dropout: Option<Value>,
    emb_layer_norm_before: bool,
    esmfold_config: Option<Value>,
    hidden_act: String,
    hidden_dropout_prob: f64,
    hidden_size: u32,
    initializer_range: f64,
    intermediate_size: u32,
    is_folding_model: bool,
    layer_norm_eps: f64,
    mask_token_id: u32,
    max_position_embeddings: u32,
    model_type: String,
    num_attention_heads: u32,
    num_hidden_layers: u32,
    pad_token_id: u32,
    position_embedding_type: String,
    token_dropout: bool,
    torch_dtype: String,
    transformers_version: String,
    use_cache: bool,
    vocab_list: Option<Value>,
    vocab_size: u32,
    type_vocab_size: u32,
    bos_token_id: u32,
    eos_token_id: u32,
}

#[test]
fn test_config() {
    let config_f = std::fs::read_to_string("data/esm2_t6_8M_UR50D/config.json").unwrap();
    let config: Config = serde_json::from_str(&config_f).unwrap();
    dbg!(config);
}

    // let config: Config = serde_json::from_str(&config)?;