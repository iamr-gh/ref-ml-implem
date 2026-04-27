package main

import "core:fmt"

// starting point is gpt 2 tokenizer
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/tokenization_gpt2.py
// https://github.com/huggingface/tokenizers

tokenize :: proc(str: string) -> [dynamic]i64 {
	// this looks pretty complicated, need to figure out the exact minimum I need

	// normalizer -- this seems to be handled already, or odin will have a solution

	// pretokenizer -- bytelevel

	// bpe model
	// -- given a vocab
	// -- given merges


	// post processing


	// byte level byte pair encoding
	return [dynamic]i64{}
}

detokenize :: proc(lst: [dynamic]i64) -> string {
	// decoder is byte level
	return ""
}

verifyTokenizer :: proc(str: string) {

	fmt.printf("Testing tok detok of {}\n", str)
	assert(detokenize(tokenize(str)) == str)
}

testTokenizer :: proc() {
	// todo: consider special tokens
	testStrings := [?]string{"abc", "Hello, I am a man of the world", "I hope to be more someday"}

	// general consistency checks (all that actually matters if I'm manually tokenizing)
	for str in testStrings {
		verifyTokenizer(str)
	}

	// specific compatibility with gpt 2 tokenizer checks
	// make sure specific tokens go to specific places
	// this is important when importing foreign datasets
	helloworld_tok := [?]i64{18435, 995}
	// assert(tokenize(" Hello world")[:] == helloworld_tok[:])
}
