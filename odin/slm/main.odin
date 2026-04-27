package main


main :: proc() {
	testTokenizer()

	// I want to target the training of a language model exactly
	// big useful thing that is my interest

	// if too compute heavy, we can go back to simpler, but big is actually valuable
	// or we can keep fighting the compute heavy
	// goal is train on local

	// input: string
	// output: string

	// nanochat architecture, proven tiny fast convergence: https://github.com/karpathy/nanochat/discussions/481

	// architecture:
	// tokenizer

	// embeddings
	// transformer layers
	// maybe some dense layers
	// unembedding

	// detokenizer

	// other things
	// data ingestion (fineweb, etc.)
	// optimizers

	// first steps
	// ingest a real pretraining dataset
	// verify can run something and learn gradients on that -- maybe an mlp that classifies a dumb property
}

// a lot of the datasets I care about are built from python integrations
// which is not inherently wrong, but it is interesting
