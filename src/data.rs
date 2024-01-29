use anyhow::{Error as E, Result};
use candle_core::{Device, DType};
use candle_core::Tensor;
// use candle_nn::VarBuilder;


use std::collections::HashMap;

// proteinseq_toks = {
//     'toks': ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-']
// }

const PROTEINSEQ_TOKS: &[&str] = &[
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-",
];
struct Alphabet {
    standard_toks: Vec<String>,
    prepend_toks: Vec<String>,
    append_toks: Vec<String>,
    prepend_bos: bool,
    append_eos: bool,
    use_msa: bool,
    all_toks: Vec<String>,
    tok_to_idx: HashMap<String, usize>,
    unk_idx: usize,
    padding_idx: usize,
    cls_idx: usize,
    mask_idx: usize,
    eos_idx: usize,
    all_special_tokens: Vec<String>,
    unique_no_split_tokens: Vec<String>,
}

impl Alphabet {
    fn new(
        standard_toks: Vec<String>,
        prepend_toks: Vec<String>,
        append_toks: Vec<String>,
        prepend_bos: bool,
        append_eos: bool,
        use_msa: bool,
    ) -> Alphabet {
        let mut all_toks = prepend_toks.clone();
        all_toks.extend(standard_toks.clone());
        // Add null tokens to make the length a multiple of 8
        let additional_nulls = (8 - (all_toks.len() % 8)) % 8;
        for i in 1..=additional_nulls {
            all_toks.push(format!("<null_{}>", i));
        }
        all_toks.extend(append_toks.clone());

        let tok_to_idx: HashMap<_, _> = all_toks.iter().enumerate().map(|(i, tok)| (tok.clone(), i)).collect();

        let unk_idx = *tok_to_idx.get("<unk>").unwrap();
        let padding_idx = tok_to_idx.get("<pad>").unwrap_or(&unk_idx).to_owned();
        let cls_idx = tok_to_idx.get("<cls>").unwrap_or(&unk_idx).to_owned();
        let mask_idx = tok_to_idx.get("<mask>").unwrap_or(&unk_idx).to_owned();
        let eos_idx = tok_to_idx.get("<eos>").unwrap_or(&unk_idx).to_owned();

        Alphabet {
            standard_toks,
            prepend_toks,
            append_toks,
            prepend_bos,
            append_eos,
            use_msa,
            all_toks: (&all_toks).clone(),
            tok_to_idx,
            unk_idx,
            padding_idx,
            cls_idx,
            mask_idx,
            eos_idx,
            all_special_tokens: vec!["<eos>".into(), "<unk>".into(), "<pad>".into(), "<cls>".into(), "<mask>".into()],
            unique_no_split_tokens: all_toks.clone(),
        }
    }

    fn len(&self) -> usize {
        self.all_toks.len()
    }

    fn get_idx(&self, tok: &str) -> usize {
        *self.tok_to_idx.get(tok).unwrap_or(&self.unk_idx)
    }

    fn get_tok(&self, ind: usize) -> &str {
        &self.all_toks[ind]
    }

    fn to_dict(&self) -> HashMap<String, usize> {
        self.tok_to_idx.clone()
    }

    // The `get_batch_converter` method will depend on the implementation of
    // `MSABatchConverter` and `BatchConverter`, which are not provided in the Python code.
    // You would need to translate or implement these classes as well in Rust.

    // The `from_architecture` method depends on external data (`proteinseq_toks`) and 
    // potentially other external dependencies. You'll need to adapt this part based on how 
    // you manage these dependencies in Rust.

    // The `tokenize` method in Python relies on dynamic string manipulation and regular expressions.
    // A direct translation to Rust would require using the `regex` crate for pattern matching
    // and implementing custom logic for splitting and tokenizing strings.
    // Below is a simplified version.

    fn tokenize(&self, text: &str) -> Vec<String> {
        // A simple placeholder implementation. You will need to adapt this method
        // based on the specific tokenization logic of your Python code.
        text.split_whitespace().map(|s| s.to_string()).collect()
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        self.tokenize(text)
            .iter()
            .map(|tok| self.get_idx(tok))
            .collect()
    }

    fn from_architecture(name: &str) -> Result<Alphabet, String> {
        let proteinseq_toks = PROTEINSEQ_TOKS.iter().map(|&x| x.to_string()).collect::<Vec<String>>();
        let (standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, use_msa) = match name {
            "ESM-1" | "protein_bert_base" => {
                // Assuming `proteinseq_toks` is a HashMap or similar type accessible here
                (proteinseq_toks, 
                 vec!["<null_0>".into(), "<pad>".into(), "<eos>".into(), "<unk>".into()], 
                 vec!["<cls>".into(), "<mask>".into(), "<sep>".into()], 
                 true, false, false)
            },
            // "ESM-1b" | "roberta_large" => {
            //     // ... similar pattern for other cases ...
            //     todo!()
            // },
            _ => return Err("Unknown architecture selected".to_owned()),
        };

        Ok(Alphabet::new(
            standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, use_msa,
        ))
    }

    // Other methods (get_idx, get_tok, to_dict, etc.) need to be implemented here.
}

// use ndarray::Array2;
// use std::vec::Vec;


struct BatchConverter {
    alphabet: Alphabet,
    truncation_seq_length: Option<usize>,
}


#[test]
fn test_from_architecture() -> Result<()>{
    let token_dict = Alphabet::from_architecture("ESM-1").unwrap().to_dict();
    let tokens = vec!["A","L", "G", "V"].into_iter().map(|x| token_dict.get(x).unwrap().to_owned()).collect::<Vec<usize>>();
    
    let tokens_true = vec![
        5,
        4,
        6,
        7,
    ];
    assert_eq!(tokens, tokens_true);
    // dbg!(tokens);
    
    Ok(())
}

#[test]
fn test_encode() {
    let alphabet = Alphabet::from_architecture("ESM-1").unwrap();
    let tokens = alphabet.encode("ALGV");
    dbg!(tokens);

}