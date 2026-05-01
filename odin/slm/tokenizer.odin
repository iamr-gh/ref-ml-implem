package main

import "core:fmt"
import "core:strings"

// starting point is gpt 2 tokenizer
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/tokenization_gpt2.py
// https://github.com/huggingface/tokenizers

tokenize :: proc(str: string) -> [dynamic]u64 {
	out := [dynamic]u64{} // note, allocation

	// this looks pretty complicated to do properly, need to figure out the exact minimum I need

	// normalizer -- this seems to be handled already, or odin will have a solution

	// pretokenizer -- bytelevel

	// bpe model
	// -- given a vocab
	// -- given merges


	// post processing -- default/not needed?


	// to verify test cases, let's just convert byte by byte
	for b in str {
		append(&out, u64(b))
	}

	// byte level byte pair encoding
	return out
}

detokenize :: proc(lst: [dynamic]u64) -> string {
	// decoder is byte level

	// placeholder string builder
	sb := strings.builder_make()
	// defer strings.builder_destroy(&sb) -- this kills the original

	for b in lst {
		strings.write_rune(&sb, rune(b))
	}

	s := strings.to_string(sb)
	return s
}

verifyTokenizer :: proc(str: string) {

	tok := tokenize(str)
	detok := detokenize(tok)

	if (detok != str) {
		fmt.printf("tok detok str of {} {} {}\n", tok, detok, str)
	}
	assert(detok == str)
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
	// helloworld_tok := [?]i64{18435, 995}
	// assert(tokenize(" Hello world")[:] == helloworld_tok[:])
}
