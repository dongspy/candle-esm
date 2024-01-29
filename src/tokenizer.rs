use candle_nn::VarBuilder;
// use tokenizer;
use candle_core::{Device, DType};
use candle_core::Tensor;
use anyhow::{Error as E, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read, BufRead};
use std::path::Path;

use rayon::vec;

const PAD_TOKEN: &str = "<pad>";
const CLS_TOKEN: &str = "<cls>";
const EOS_TOKEN: &str = "<eos>";
const UNK_TOKEN: &str = "<unk>";

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

#[derive(Clone, Debug)]
pub struct EsmTokenizer{
    id_to_token: HashMap<u32, String>,
    token_to_id: HashMap<String, u32>

}

impl EsmTokenizer{
    pub fn load_vocab_file(vocab_file: &str) -> Self{
        let mut id_to_token = HashMap::new();
        let mut token_to_id = HashMap::new();
        if let Ok(lines) = read_lines(vocab_file) {
            // Consumes the iterator, returns an (Optional) String
            for (idx, line) in lines.enumerate() {
                if let Ok(token) = line {
                    // println!("{}", ip);
                    id_to_token.insert(idx as u32, token.clone());
                    token_to_id.insert(token, idx as u32);

                }
            }
        }
    
        EsmTokenizer{
            id_to_token,
            token_to_id
        }
    }

    pub fn encode(&self, token: &str) -> u32{
        self.token_to_id[token]
    }

    pub fn batch_encode(&self, tokens: &[String]) -> Vec<u32> {
        let n_tokens = tokens.len();
        let mut ids:Vec<u32> = Vec::with_capacity(n_tokens + 1);
        ids.push(self.encode(CLS_TOKEN));
        tokens.iter().for_each(|x| ids.push(self.encode(x)));
        ids.push(self.encode(EOS_TOKEN));
        ids
    }
}

#[test]
fn test_load_vocab_file(){
    let tokenizers = EsmTokenizer::load_vocab_file("/Users/pidong/learn/DL/huggingface/esm-1b/vocab.txt");
    let tokens = vec!["A","L", "G", "V"].into_iter().map(|x| x.to_string()).collect::<Vec<String>>();
    let ids = tokenizers.batch_encode(&tokens);
    dbg!(ids);
    // dbg!(tokenizers);
}

#[test]
fn build_bert() -> Result<()>{
    let device = Device::Cpu;
    let dtype = DType::F32;
    // let config = std::fs::read_to_string("data/esm2_t6_8M_UR50D/esm2_t6_8M_UR50D/config_candle.json")?;
    // let config: Config = serde_json::from_str(&config)?;
    // let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let tokenizers = EsmTokenizer::load_vocab_file("data/esm2_t6_8M_UR50D/esm2_t6_8M_UR50D/vocab.txt");
    let tokens = vec!["A","L", "G", "V"].into_iter().map(|x| x.to_string()).collect::<Vec<String>>();
    let ids = tokenizers.batch_encode(&tokens);
    let token_ids = Tensor::new(&ids[..], &device)?.unsqueeze(0)?;
    let token_type_ids = token_ids.zeros_like()?;
    println!("token_ids: {token_ids}");
    // return Ok(());
    let vb = VarBuilder::from_pth(
        "data/esm2_t6_8M_UR50D/esm2_t6_8M_UR50D/pytorch_model.bin", dtype, &device)?;
        // unsafe { VarBuilder::from_mmaped_safetensors(&["/Users/pidong/learn/DL/huggingface/esm2_t6_8M_UR50D/model_without_esm.safetensors"], DTYPE, &device)? };
    // VarBuilder::from_pth(p, dtype, dev)
    // let model = BertModel::load(vb, &config)?;
    // vb.pp(s)
    // dbg!(vb);
    Ok(())
}